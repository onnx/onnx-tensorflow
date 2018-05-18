from .conv_common import ConvCommon


class Convolution(ConvCommon):
  _TF_OP = ["Conv1d", "Conv2d", "Conv3d"]
  _ONNX_OP = "Conv"

  @classmethod
  def version_1(cls, node, **kwargs):
    if node.op == "Conv1d":
      d = 1
    elif node.op == "Conv2d":
      d = 2
    elif node.op == "Conv3d":
      d = 3
    else:
      raise RuntimeError("{} is not supported.".format(node.op))
    return cls.conv_op(node, 1, d=d, **kwargs)
