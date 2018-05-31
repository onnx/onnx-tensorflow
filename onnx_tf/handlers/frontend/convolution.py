from onnx_tf.common import exception
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op
from .conv_mixin import ConvMixin


@onnx_op("Conv")
@tf_op(["Conv1D", "Conv2D", "Conv3D"])
class Convolution(ConvMixin, FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    if node.op_type == "Conv1D":
      d = 1
    elif node.op_type == "Conv2D":
      d = 2
    elif node.op_type == "Conv3D":
      d = 3
    else:
      exception.OP_UNSUPPORTED_EXCEPT(node.op_type, "Tensorflow")
    return cls.conv_op(node, d=d, **kwargs)
