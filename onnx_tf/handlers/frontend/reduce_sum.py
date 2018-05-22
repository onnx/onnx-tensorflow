from onnx_tf.handlers.frontend_handler import FrontendHandler
from .math_mixin import ReductionMixin


class ReduceSum(ReductionMixin, FrontendHandler):
  TF_OP = ["Sum"]
  ONNX_OP = "ReduceSum"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.reduction_op(node, **kwargs)
