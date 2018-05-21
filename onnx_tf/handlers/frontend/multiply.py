from .math_common import ArithmeticCommon


class Multiply(ArithmeticCommon):
  TF_OP = ["Mul"]
  ONNX_OP = "Mul"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.arithmetic_op(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.arithmetic_op(node, **kwargs)
