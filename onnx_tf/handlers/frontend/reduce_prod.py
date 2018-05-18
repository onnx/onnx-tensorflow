from .math_common import ReductionCommon


class ReduceProd(ReductionCommon):
  _TF_OP = ["Prod"]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.reduction_op(node, 1, **kwargs)
