import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("Expand")
class Expand(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x, shape = tensor_dict[node.inputs[0]], tensor_dict[node.inputs[1]]

    # tf.math.multiply does not support bool therefore use int8
    if x.dtype is tf.bool:
      ones = tf.ones(shape, dtype=tf.int8)
      r = tf.cast(x, tf.int8) * ones
      return [tf.cast(r, tf.bool)]
    else:
      ones = tf.ones(shape, dtype=x.dtype)
      return [x * ones]

  @classmethod
  def version_8(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
