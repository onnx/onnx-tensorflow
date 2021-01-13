import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from .math_mixin import BasicMathMixin


@onnx_op("Reciprocal")
@tf_func(tf.math.reciprocal)
class Reciprocal(BasicMathMixin, BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
