from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.frontend_handler import version
from .math_mixin import BasicMathMixin


class Asin(BasicMathMixin, FrontendHandler):

  @classmethod
  @version(7)
  def version_7(cls, node, **kwargs):
    return cls.basic_math_op(node, **kwargs)
