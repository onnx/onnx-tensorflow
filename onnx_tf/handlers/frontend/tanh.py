from onnx_tf.handlers.frontend_handler import FrontendHandler
from .math_mixin import BasicMathMixin


class Tanh(BasicMathMixin, FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.basic_math_op(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.basic_math_op(node, **kwargs)
