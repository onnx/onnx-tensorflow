from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op
from .math_mixin import BasicMathMixin


@onnx_op("Atanh")
@tf_op("Atanh")
class Atanh(BasicMathMixin, FrontendHandler):

  @classmethod
  def version_9(cls, node, **kwargs):
    return cls.basic_math_op(node, **kwargs)
