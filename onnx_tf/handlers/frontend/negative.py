from .math_common import BasicMathCommon


class Negative(BasicMathCommon):
  _TF_OP = ["Neg"]
  _ONNX_OP = "Neg"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.basic_math_op(node, 1)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.basic_math_op(node, 6)
