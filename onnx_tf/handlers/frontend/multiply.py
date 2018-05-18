from .math_common import ArithmeticCommon


class Multiply(ArithmeticCommon):
  _TF_OP = ["Mul"]
  _ONNX_OP = "Mul"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.arithmetic_op(node, 1)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.arithmetic_op(node, 6)
