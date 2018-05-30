from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op
from .math_mixin import ArithmeticMixin


@onnx_op("Add")
@tf_op("Add")
class Add(ArithmeticMixin, FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.arithmetic_op(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.arithmetic_op(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls.arithmetic_op(node, **kwargs)