import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("ArgMax")
@tf_func(tf.argmax)
class ArgMax(BackendHandler):

  @classmethod
  def get_attrs_processor_param(cls):
    return {"default": {"axis": 0}}

  @classmethod
  def version_1(cls, node, **kwargs):
    axis = node.attrs.get("axis", 0)
    keepdims = node.attrs.get("keepdims", 1)
    arg_max = cls.make_tensor_from_onnx_node(node, **kwargs)
    if keepdims == 1:
      return [tf.expand_dims(arg_max, axis=axis)]
    return [arg_max]
