from onnx_tf.handlers.frontend_handler import FrontendHandler
from .control_flow_mixin import LogicalMixin


class LogicalOr(LogicalMixin, FrontendHandler):
  ONNX_OP = "Or"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.logical_op(node, **kwargs)
