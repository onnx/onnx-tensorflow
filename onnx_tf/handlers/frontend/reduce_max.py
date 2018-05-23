from onnx_tf.handlers.frontend_handler import FrontendHandler
from .math_mixin import ReductionMixin


class ReduceMax(ReductionMixin, FrontendHandler):
  TF_OP = ["Max"]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.reduction_op(node, **kwargs)
