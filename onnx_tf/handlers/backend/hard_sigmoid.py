import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("HardSigmoid")
class HardSigmoid(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    if "alpha" not in node.attrs and "beta" not in node.attrs:
      return [tf.keras.backend.hard_sigmoid(x)]

    alpha = node.attrs.get("alpha", 0.2)
    beta = node.attrs.get("beta", 0.5)
    return [tf.clip_by_value(x * alpha + beta, 0, 1)]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls._common(node, **kwargs)
