import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from .math_mixin import ArithmeticMixin


@onnx_op("Div")
@tf_func(tf.math.truediv)
class Div(ArithmeticMixin, BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    # Tensorflow automatically casts inputs to a float type for integers
    # in tf.math.truediv and returns output in floating point. We need to
    # case the output back to the original data type when needed.
    dtype = kwargs['tensor_dict'][node.inputs[0]].dtype
    result = [cls.make_tensor_from_onnx_node(node, **kwargs)] if dtype in [
        tf.float16, tf.float32, tf.float64, tf.bfloat16
    ] else [tf.cast(cls.make_tensor_from_onnx_node(node, **kwargs), dtype)]
    return result

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.limited_broadcast(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.limited_broadcast(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_14(cls, node, **kwargs):
    return cls._common(node, **kwargs)
