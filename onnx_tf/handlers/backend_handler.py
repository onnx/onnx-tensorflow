from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import inspect
import sys

import tensorflow as tf

from onnx_tf.common import get_data_format
from onnx_tf.common import get_perm_from_formats
from onnx_tf.common import supports_device
from .handler import Handler


class BackendHandler(Handler):
  """ This class is base backend handler class.
  All backend operator handler class MUST inherit this class.
  In backend, operator handler class's name should be pascal case of file name
  which should be snake case.
  Use ONNX operator name as class name.
  """

  TF_FUNC = None

  @classmethod
  def get_attrs_processor_param(cls):
    """ Get param for attrs processor.

    :return: Dict.
    """
    return {}

  @classmethod
  def _process_attrs(cls, attrs):
    """ Private method for processing attrs.
    Param for this processor got from `get_attrs_processor_param`.
    Param is dict contains two key: `default` and `raname`.
    First add default value to attrs if key does not exist.
    Second rename key to new key.

    For example:
      attrs = {"keep_dims": True}
      param = {"default": {"axis": 1},
               "rename": {"keep_dims": "keepdims"}}

      processed_attrs = {"axis": "1", "keepdims": True}

    :param attrs: Process target attrs.
    :return: Processed attrs.
    """
    param = {"rename": {}, "default": {}}
    param.update(cls.get_attrs_processor_param())

    for k, v in param["default"].items():
      attrs.setdefault(k, v)

    for k, new_k in param["rename"].items():
      if k in attrs:
        attrs[new_k] = attrs.pop(k)

    return attrs

  @classmethod
  def make_tensor_from_onnx_node(cls,
                                 node,
                                 tf_func=None,
                                 inputs=None,
                                 attrs=None,
                                 name=None,
                                 c_first_cuda_only=False,
                                 c_last_only=False,
                                 **kwargs):
    """ Helper method to make tensor.

    :param node: OnnxNode object.
    :param tf_func: Tf function. Default is cls.TF_FUNC.
    :param inputs: Inputs tensor. Default is got from node.inputs.
    :param attrs: Attributes. Default is processed node.attrs.
    :param name: Node name.
    :param c_first_cuda_only: If channel first is only supported by cuda.
    If true and not cuda, do pre and post transpose.
    :param c_last_only: If only channel last is support,
    do pre and post transpose.
    :param kwargs: Other args.
    :return: Tensor.
    """
    tensor_dict = kwargs.get("tensor_dict", {})
    tf_func = tf_func or cls.TF_FUNC
    inputs = inputs or [tensor_dict.get(inp, None) for inp in node.inputs]
    attrs = cls._process_attrs(attrs or copy.deepcopy(node.attrs))
    name = name or node.name
    if name != "":
      attrs["name"] = name

    if not c_first_cuda_only and not c_last_only:
      return cls._run_tf_func(tf_func, inputs, attrs)
    else:
      support_cuda = supports_device("CUDA")
      x = inputs[0]
      x_rank = len(x.get_shape())
      storage_format, compute_format = get_data_format(x_rank)
      pre_perm = list(range(x_rank))
      post_perm = pre_perm[:]

      if c_first_cuda_only and not support_cuda:
        pre_perm = get_perm_from_formats(storage_format, compute_format)
        post_perm = get_perm_from_formats(compute_format, storage_format)
      if c_last_only:
        compute_format = compute_format.replace("C", "") + "C"
        pre_perm = get_perm_from_formats(storage_format, compute_format)
        post_perm = get_perm_from_formats(compute_format, storage_format)

      attrs["data_format"] = compute_format

      if pre_perm != list(range(x_rank)):
        x_t = tf.transpose(x, perm=pre_perm)
        y = cls._run_tf_func(tf_func, [x_t] + inputs[1:], attrs)
        y_t = tf.transpose(y, perm=post_perm)
        return y_t

      return cls._run_tf_func(tf_func, inputs, attrs)

  @classmethod
  def _run_tf_func(cls, tf_func, inputs, attrs):
    """ Run Tensorflow function.
    Use acceptable attrs of function from attrs only.

    :param tf_func: Tensorflow function.
    :param inputs: Inputs.
    :param attrs: Attributes.
    :return: Tensor.
    """
    if sys.version_info > (3,):
      params = list(inspect.signature(tf_func).parameters.keys())
    else:
      # use closure to get args for function using decorator
      if tf_func.__closure__ is not None:
        params = tf_func.__closure__[1].cell_contents.args
      else:
        params = inspect.getargspec(tf_func).args

    return tf_func(*inputs,
                   **dict([(p, attrs[p]) for p in params if p in attrs]))
