from .control_flow_common import LogicalCommon


class LogicalXor(LogicalCommon):
  ONNX_OP = "Xor"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.logical_op(node, 1)
