from onnx_tf.common import exception
from onnx_tf.handlers.frontend_handler import FrontendHandler
from .conv_mixin import ConvMixin


class Convolution(ConvMixin, FrontendHandler):
  TF_OP = ["Conv1D", "Conv2D", "Conv3D"]
  ONNX_OP = "Conv"

  @classmethod
  def version_1(cls, node, **kwargs):
    if node.op == "Conv1D":
      d = 1
    elif node.op == "Conv2D":
      d = 2
    elif node.op == "Conv3D":
      d = 3
    else:
      exception.OP_UNSUPPORTED_EXCEPT(node.op, "Tensorflow")
    return cls.conv_op(node, d=d, **kwargs)
