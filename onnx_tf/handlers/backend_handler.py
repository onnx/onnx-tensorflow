from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import inspect

import tensorflow as tf

from onnx_tf.common import IS_PYTHON3
from onnx_tf.common import get_data_format
from onnx_tf.common import get_perm_from_formats
from onnx_tf.common import sys_config
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
  def get_req_vars_template(cls, node, init_dict):
    """ Get required variables template, which is a
    dictionary of variable names with initial value and
    shape
    :param node: ONNX NodeProto object.
    :param init_dict: initializer dictionary of the graph.
    :return: Dictionary.
    """
    return {}

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
                                 name="",
                                 c_first_cuda_only=False,
                                 c_last_only=False,
                                 **kwargs):
    """ Helper method to make tensor.

    :param node: OnnxNode object.
    :param tf_func: Callable Tf function. Default is cls.TF_FUNC.
    :param inputs: Inputs tensor. Default is got from node.inputs.
    :param attrs: Attributes. Default is node.attrs.
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
    if tf_func is None:
      raise RuntimeError("No Tensorflow function is given.")
    if inputs is None:
      inputs = [tensor_dict.get(inp, None) for inp in node.inputs]
    if attrs is None:
      attrs = copy.deepcopy(node.attrs)
    name = name or node.name
    if name != "":
      attrs["name"] = name

    if c_first_cuda_only and c_last_only:
      raise ValueError(
          "c_first_cuda_only and c_last_only can not both be True.")

    if c_first_cuda_only:
      return cls.c_first_cuda_only(tf_func, inputs, attrs)
    elif c_last_only:
      return cls.c_last_only(tf_func, inputs, attrs)

    return cls._run_tf_func(tf_func, inputs, attrs)

  @classmethod
  def c_first_cuda_only(cls, tf_func, inputs, attrs):
    """ Handle operator that channel first is only supported by CUDA.
    When using CPU, two transposes should be added.

    :param tf_func: Callable Tf function.
    :param inputs: Inputs tensor.
    :param attrs: Attributes.
    :return: Tensor.
    """
    if sys_config.device == 'CPU':
      return cls._tuck_transpose(tf_func, inputs, attrs)
    return cls._run_tf_func(tf_func, inputs, attrs)

  @classmethod
  def c_last_only(cls, tf_func, inputs, attrs):
    """ Handle operator that channel last only is supported.
    Add two transposes anyway.

    :param tf_func: Callable Tf function.
    :param inputs: Inputs tensor.
    :param attrs: Attributes.
    :return: Tensor.
    """
    storage_format, compute_format = get_data_format(len(inputs[0].get_shape()))
    compute_format = compute_format.replace("C", "") + "C"
    return cls._tuck_transpose(tf_func, inputs, attrs,
                               (storage_format, compute_format))

  @classmethod
  def _tuck_transpose(cls, tf_func, inputs, attrs, data_format=None):
    x = inputs[0]
    x_rank = len(x.get_shape())
    if not data_format:
      data_format = get_data_format(x_rank)
    pre_perm = get_perm_from_formats(data_format[0], data_format[1])
    post_perm = get_perm_from_formats(data_format[1], data_format[0])
    attrs["data_format"] = data_format[1]
    if pre_perm != list(range(x_rank)):
      x_t = tf.transpose(x, perm=pre_perm)
      y = cls._run_tf_func(tf_func, [x_t] + inputs[1:], attrs)
      y_t = tf.transpose(y, perm=post_perm)
      return y_t
    return cls._run_tf_func(tf_func, inputs, attrs)

  @classmethod
  def _run_tf_func(cls, tf_func, inputs, attrs):
    """ Run Tensorflow function.
    Use only acceptable attributes of function from attrs.

    :param tf_func: Tensorflow function.
    :param inputs: Inputs.
    :param attrs: Attributes.
    :return: Tensor.
    """
    if IS_PYTHON3:
      params = list(inspect.signature(tf_func).parameters.keys())
    else:
      # use closure to get args for function using decorator
      if tf_func.__closure__ is not None:
        while "__wrapped__" in tf_func.func_dict:
          tf_func = tf_func.func_dict["__wrapped__"]
        params = inspect.getargspec(tf_func).args
      else:
        params = inspect.getargspec(tf_func).args

    attrs = cls._process_attrs(attrs)

    if "name" in attrs.keys():
      attrs["name"] = "onnx_tf_prefix_" + attrs["name"]

    attrs = {p: v for p, v in attrs.items() if p in params}
    kwargs = dict(zip(params, inputs))
    ambiguous_arguments = any(
        kwargs.get(p) is not None and v is not None for p, v in attrs.items())
    if ambiguous_arguments:
      raise TypeError('Ambiguous arguments for {}()'.format(tf_func.__name__))
    kwargs.update((p, v) for p, v in attrs.items() if v is not None)
    return tf_func(**kwargs)
