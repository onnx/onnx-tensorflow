from .math_common import ReductionCommon


class ReduceMax(ReductionCommon):
  TF_OP = ["Max"]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.reduction_op(node, 1, **kwargs)
