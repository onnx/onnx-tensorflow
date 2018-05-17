from .control_flow_common import LogicalCommon


class LogicalNot(LogicalCommon):
  _ONNX_OP = "Not"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.logical_op(node, 1, broadcast=0)
