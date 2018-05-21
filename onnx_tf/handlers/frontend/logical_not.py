from .control_flow_common import LogicalCommon


class LogicalNot(LogicalCommon):
  ONNX_OP = "Not"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.logical_op(node, broadcast=0, **kwargs)
