from onnx_tf.handlers.frontend_handler import FrontendHandler
from .math_mixin import ArithmeticMixin


class Div(ArithmeticMixin, FrontendHandler):
  TF_OP = ["RealDiv"]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.arithmetic_op(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.arithmetic_op(node, **kwargs)
