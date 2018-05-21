from .math_common import BasicMathCommon


class Negative(BasicMathCommon):
  TF_OP = ["Neg"]
  ONNX_OP = "Neg"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.basic_math_op(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.basic_math_op(node, **kwargs)
