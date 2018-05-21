from .control_flow_common import LogicalCommon


class LogicalOr(LogicalCommon):
  ONNX_OP = "Or"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.logical_op(node, **kwargs)
