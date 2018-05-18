from .math_common import ReductionCommon


class ReduceSum(ReductionCommon):
  _TF_OP = ["Sum"]
  _ONNX_OP = "ReduceSum"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.reduction_op(node, 1, **kwargs)
