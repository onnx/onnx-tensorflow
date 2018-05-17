from .math_common import ReductionCommon


class Max(ReductionCommon):
  _ONNX_OP = "ReduceMax"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.reduction_op(node, 1, **kwargs)
