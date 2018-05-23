from onnx_tf.handlers.frontend_handler import FrontendHandler
from .math_mixin import BasicMathMixin


class Atan(BasicMathMixin, FrontendHandler):

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls.basic_math_op(node, **kwargs)
