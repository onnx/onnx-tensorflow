import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("Gemm")
class Gemm(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]
    x = tf.keras.layers.Flatten()(x)
    y = tensor_dict[node.inputs[1]]

    if len(node.inputs) > 2:
      z = tensor_dict[node.inputs[2]]
    else:
      z = 0

    if node.attrs.get("transA", 0):
      x = tf.transpose(x)
    if node.attrs.get("transB", 0):
      y = tf.transpose(y)
    alpha = node.attrs.get("alpha", 1.0)
    beta = node.attrs.get("beta", 1.0)

    return [alpha * tf.matmul(x, y) + beta * z]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_9(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)
