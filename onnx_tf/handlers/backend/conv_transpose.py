from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op
from .conv_mixin import ConvMixin


@onnx_op("ConvTranspose")
@tf_op(["Conv2DBackpropInput", "Conv3DBackpropInput"])
class ConvTranspose(ConvMixin, BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.conv(node, kwargs["tensor_dict"], transpose=True)
