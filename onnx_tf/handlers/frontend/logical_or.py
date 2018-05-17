from .control_flow_common import LogicalCommon


class LogicalOr(LogicalCommon):
  _ONNX_OP = "Or"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.logical_op(node, 1)
