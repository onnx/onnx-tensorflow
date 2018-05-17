from .math_common import ReductionCommon


class Mean(ReductionCommon):
  _ONNX_OP = "ReduceMean"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.reduction_op(node, 1, **kwargs)
