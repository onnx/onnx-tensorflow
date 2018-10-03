from onnx_tf.common import exception
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op
from .conv_mixin import ConvMixin


@onnx_op("Conv")
@tf_op(["Conv1D", "Conv2D", "Conv3D", "DepthwiseConv2dNative"])
class Convolution(ConvMixin, FrontendHandler):
  """
    Converts different convolutions
    Warning: Depthwise Conv is not supported by ONNX directly, so this generates
    a grouped convolution with n_groups = n_channels which is semantically the same.
    Make sure your backend knows about this special case in order
    to generate more optimal code.
  """

  @classmethod
  def version_1(cls, node, **kwargs):
    if node.op_type == "Conv1D":
      d = 1
    elif node.op_type == "Conv2D":
      d = 2
    elif node.op_type == "Conv3D":
      d = 3
    elif node.op_type == "DepthwiseConv2dNative":
      d = 2
      return cls.conv_op(node, d=d, is_depthwise=True, **kwargs)
    else:
      exception.OP_UNSUPPORTED_EXCEPT(node.op_type, "Tensorflow")
    return cls.conv_op(node, d=d, **kwargs)
