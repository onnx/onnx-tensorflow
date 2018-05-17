from .math_common import ReductionCommon


class Min(ReductionCommon):
  _ONNX_OP = "ReduceMin"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.reduction_op(node, 1, **kwargs)
