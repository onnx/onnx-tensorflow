from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.frontend_handler import version
from .math_mixin import ReductionMixin


class ReduceSum(ReductionMixin, FrontendHandler):
  TF_OP = ["Sum"]
  ONNX_OP = "ReduceSum"

  @classmethod
  @version(1)
  def version_1(cls, node, **kwargs):
    return cls.reduction_op(node, **kwargs)
