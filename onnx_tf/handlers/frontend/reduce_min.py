from onnx_tf.handlers.frontend_handler import FrontendHandler
from .math_mixin import ReductionMixin


class ReduceMin(ReductionMixin, FrontendHandler):
  TF_OP = ["Min"]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.reduction_op(node, **kwargs)
