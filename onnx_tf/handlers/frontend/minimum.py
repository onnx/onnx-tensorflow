from .math_common import BasicMathCommon


class Minimum(BasicMathCommon):
  ONNX_OP = "Min"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.basic_math_op(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.basic_math_op(node, **kwargs)
