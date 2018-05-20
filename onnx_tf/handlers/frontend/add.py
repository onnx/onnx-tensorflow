from .math_common import ArithmeticCommon


class Add(ArithmeticCommon):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.arithmetic_op(node, 1)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.arithmetic_op(node, 6)