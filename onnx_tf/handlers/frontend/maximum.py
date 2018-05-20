from .math_common import BasicMathCommon


class Maximun(BasicMathCommon):
  _ONNX_OP = "Max"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.basic_math_op(node, 1)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.basic_math_op(node, 6)