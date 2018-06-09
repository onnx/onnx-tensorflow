import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Elu")
@tf_func(tf.nn.elu)
class Elu(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    alpha = node.attrs.get("alpha", 1.0)
    if alpha != 1.0:
      return [
          tf.cast(x < 0.0, tf.float32) * alpha *
          (tf.exp(x) - 1.0) + tf.cast(x >= 0.0, tf.float32) * x
      ]
    else:
      return [cls.make_tensor_from_onnx_node(node, **kwargs)]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls._common(node, **kwargs)
