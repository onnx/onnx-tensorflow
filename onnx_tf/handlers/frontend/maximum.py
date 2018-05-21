from .math_common import BasicMathCommon


class Maximun(BasicMathCommon):
  ONNX_OP = "Max"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.basic_math_op(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.basic_math_op(node, **kwargs)
