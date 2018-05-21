from .math_common import ReductionCommon


class ReduceMean(ReductionCommon):
  TF_OP = ["Mean"]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.reduction_op(node, 1, **kwargs)
