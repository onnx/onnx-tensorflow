import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("ThresholdedRelu")
class ThresholdedRelu(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    alpha = node.attrs.get("alpha", 1.0)
    epsilon = 1e-5
    return [tf.nn.relu(x) - tf.nn.relu(tf.sign(alpha - x + epsilon) * x)]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_10(cls, node, **kwargs):
    return cls._common(node, **kwargs)
