from .math_common import ArithmeticCommon


class Subtract(ArithmeticCommon):
  _TF_OP = ["Sub"]
  _ONNX_OP = "Sub"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.arithmetic_op(node, 1)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.arithmetic_op(node, 6)
