from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.frontend_handler import version
from .math_mixin import BasicMathMixin


class Negative(BasicMathMixin, FrontendHandler):
  TF_OP = ["Neg"]
  ONNX_OP = "Neg"

  @classmethod
  @version(1)
  def version_1(cls, node, **kwargs):
    return cls.basic_math_op(node, **kwargs)

  @classmethod
  @version(6)
  def version_6(cls, node, **kwargs):
    return cls.basic_math_op(node, **kwargs)
