from .math_common import ReductionCommon


class Prod(ReductionCommon):
  _ONNX_OP = "ReduceProd"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.reduction_op(node, 1, **kwargs)
