from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op
from .math_mixin import BasicMathMixin


@onnx_op("Log")
@tf_op("Log")
class Log(BasicMathMixin, FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.basic_math_op(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.basic_math_op(node, **kwargs)
