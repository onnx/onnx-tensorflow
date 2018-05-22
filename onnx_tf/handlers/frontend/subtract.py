from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.frontend_handler import version
from .math_mixin import ArithmeticMixin


class Subtract(ArithmeticMixin, FrontendHandler):
  TF_OP = ["Sub"]
  ONNX_OP = "Sub"

  @classmethod
  @version(1)
  def version_1(cls, node, **kwargs):
    return cls.arithmetic_op(node, **kwargs)

  @classmethod
  @version(6)
  def version_6(cls, node, **kwargs):
    return cls.arithmetic_op(node, **kwargs)
