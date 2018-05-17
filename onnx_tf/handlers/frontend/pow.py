from .math_common import BasicMathCommon


class Pow(BasicMathCommon):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.basic_math_op(node, 1)
