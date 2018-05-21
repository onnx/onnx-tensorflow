from .math_common import BasicMathCommon


class Atan(BasicMathCommon):

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls.basic_math_op(node, **kwargs)
