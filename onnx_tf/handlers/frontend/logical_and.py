from .control_flow_common import LogicalCommon


class LogicalAnd(LogicalCommon):
  _ONNX_OP = "And"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.logical_op(node, 1)
