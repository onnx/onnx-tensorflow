from .math_common import ReductionCommon


class ReduceMin(ReductionCommon):
  _TF_OP = ["Min"]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.reduction_op(node, 1, **kwargs)