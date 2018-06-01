from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy

import tensorflow as tf

from onnx_tf.common import get_data_format
from onnx_tf.common import get_perm_from_formats
from onnx_tf.common import supports_device
from .handler import Handler


# TODO(fumihwh): fix all doc
class BackendHandler(Handler):
  """ This class is base frontend handler class.
  All frontend operator handler class MUST inherit this class.
  In frontend, operator handler class's name should be pascal case of file name
  which should be snake case.
  It is best to use tf functions' names. e.g. tf.nn.avg_pool
  If there is a multiple mapping, e.g. tf.nn.conv1d, tf.nn.conv2d, tf.nn.conv3d,
  try find common one first. In this case, tf.nn.convolution.
  Finally, use ONNX name if above does not work.
  """

  TF_FUNC = None

  @classmethod
  def process_attrs(cls, attrs):
    return attrs

  @classmethod
  def _process_attrs(cls, attrs, remove=None, rename=None, default=None):
    remove = remove or []
    for k in remove:
      attrs.pop(k, None)

    default = default or {}
    for k, v in default.items():
      attrs.setdefault(k, v)

    rename = rename or {}
    for k, new_k in rename.items():
      if k in attrs:
        attrs[new_k] = attrs.pop(k)

    return attrs

  # @classmethod
  # def check_cls(cls):
  #   super(BackendHandler, cls).check_cls()
  #   if not cls.TF_FUNC:
  #     raise ValueError(
  #         "{} doesn't have TF_FUNC. "
  #         "Please use Handler.tf_func decorator to register TF_FUNC.".format(
  #             cls.__name__))

  @classmethod
  def make_tf_tensor(cls,
                     node,
                     tf_func=None,
                     inputs=None,
                     attrs=None,
                     name=None,
                     c_first_cuda_only=False,
                     c_last_only=False,
                     **kwargs):
    tensor_dict = kwargs.get("tensor_dict", {})
    tf_func = tf_func or cls.TF_FUNC
    inputs = inputs or [tensor_dict.get(inp, None) for inp in node.inputs]
    attrs = cls.process_attrs(attrs or copy.deepcopy(node.attrs))
    name = name or node.name
    if name != "":
      attrs["name"] = name

    if not c_first_cuda_only and not c_last_only:
      return tf_func(*inputs, **attrs)
    else:
      support_cuda = supports_device("CUDA")
      x = inputs[0]
      storage_format, compute_format = get_data_format(
          len(x.get_shape()), support_cuda)
      pre_perm = list(range(len(x.get_shape())))
      post_perm = pre_perm[:]

      if c_first_cuda_only and not support_cuda:
        pre_perm = get_perm_from_formats(storage_format, compute_format)
        post_perm = get_perm_from_formats(compute_format, storage_format)
        attrs["data_format"] = compute_format
      if c_last_only:
        compute_format = compute_format.replace("C", "") + "C"
        pre_perm = get_perm_from_formats(storage_format, compute_format)
        post_perm = get_perm_from_formats(compute_format, storage_format)

      if pre_perm != list(range(len(x.get_shape()))):
        x_t = tf.transpose(x, perm=pre_perm)
        inputs[0] = x_t
        y = tf_func(*inputs, **attrs)
        y_t = tf.transpose(y, perm=post_perm)
        return y_t

      return tf_func(*inputs, **attrs)
