from .math_common import ReductionCommon


class Sum(ReductionCommon):
  _ONNX_OP = "ReduceSum"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.reduction_op(node, 1, **kwargs)
