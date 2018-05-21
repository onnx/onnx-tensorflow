from .math_common import ArithmeticCommon


class Subtract(ArithmeticCommon):
  TF_OP = ["Sub"]
  ONNX_OP = "Sub"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.arithmetic_op(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.arithmetic_op(node, **kwargs)
