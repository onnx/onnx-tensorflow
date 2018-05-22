from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.frontend_handler import version
from .math_mixin import ArithmeticMixin


class BiasAdd(ArithmeticMixin, FrontendHandler):
  ONNX_OP = "Add"

  @classmethod
  @version(1)
  def version_1(cls, node, **kwargs):
    data_format = node.attr.get("data_format", "NHWC")
    channel_first = data_format[1] == "C"
    axis = 1 if channel_first else -1
    return cls.arithmetic_op(node, axis=axis, **kwargs)

  @classmethod
  @version(6)
  def version_6(cls, node, **kwargs):
    data_format = node.attr.get("data_format", "NHWC")
    channel_first = data_format[1] == "C"
    axis = 1 if channel_first else -1
    return cls.arithmetic_op(node, axis=axis, **kwargs)
