from .math_common import ArithmeticCommon


class BiasAdd(ArithmeticCommon):
  _ONNX_OP = "Add"

  @classmethod
  def version_1(cls, node, **kwargs):
    data_format = node.attr.get("data_format", "NHWC")
    channel_first = data_format[1] == "C"
    axis = 1 if channel_first else -1
    return cls.arithmetic_op(node, 1, axis=axis)

  @classmethod
  def version_6(cls, node, **kwargs):
    data_format = node.attr.get("data_format", "NHWC")
    channel_first = data_format[1] == "C"
    axis = 1 if channel_first else -1
    return cls.arithmetic_op(node, 6, axis=axis)
