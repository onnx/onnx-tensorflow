import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.common.tf_helper import tf_shape


@onnx_op("ArgMin")
@tf_func(tf.argmin)
class ArgMin(BackendHandler):

  @classmethod
  def get_attrs_processor_param(cls):
    return {"default": {"axis": 0}}

  @classmethod
  def _common(cls, node, **kwargs):
    axis = node.attrs.get("axis", 0)
    keepdims = node.attrs.get("keepdims", 1)
    select_last_index = node.attrs.get("select_last_index", 0)
    if select_last_index == 0:
      arg_min = cls.make_tensor_from_onnx_node(node, **kwargs)
    else:
      # reverse the input and apply argmax on that to get last occurrence of max
      x = kwargs["tensor_dict"][node.inputs[0]]
      x = tf.reverse(x, axis=[axis])
      arg_min = cls.make_tensor_from_onnx_node(node, inputs=[x], **kwargs)
      # adjust indices to account for the reverse
      arg_min = tf_shape(x)[axis] - arg_min - 1
    if keepdims == 1:
      return [tf.expand_dims(arg_min, axis=axis)]
    return [arg_min]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_12(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
