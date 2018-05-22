from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.frontend_handler import version
from .control_flow_mixin import LogicalMixin


class LogicalNot(LogicalMixin, FrontendHandler):
  ONNX_OP = "Not"

  @classmethod
  @version(1)
  def version_1(cls, node, **kwargs):
    return cls.logical_op(node, broadcast=0, **kwargs)
