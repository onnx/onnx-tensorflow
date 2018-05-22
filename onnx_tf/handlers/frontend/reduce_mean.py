from onnx_tf.handlers.frontend_handler import FrontendHandler
from .math_mixin import ReductionMixin


class ReduceMean(ReductionMixin, FrontendHandler):
  TF_OP = ["Mean"]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.reduction_op(node, **kwargs)
