import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("NonZero")
class NonZero(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    input_tensor = kwargs["tensor_dict"][node.inputs[0]]
    condition = tf.not_equal(input_tensor, tf.zeros_like(input_tensor))
    nonzero_indices = tf.where(condition)
    return [tf.transpose(nonzero_indices)]

  @classmethod
  def version_9(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
