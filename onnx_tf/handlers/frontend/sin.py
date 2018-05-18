from .math_common import BasicMathCommon


class Sin(BasicMathCommon):

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls.basic_math_op(node, 7)
