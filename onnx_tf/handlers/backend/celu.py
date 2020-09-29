import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("Celu")
class Celu(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    alpha = node.attrs.get("alpha", 1.0)

    return [
        tf.cast(x < 0.0, tf.float32) * alpha * (tf.exp(x / alpha) - 1.0) +
        tf.cast(x >= 0.0, tf.float32) * x
    ]

  @classmethod
  def version_12(cls, node, **kwargs):
    return cls._common(node, **kwargs)
