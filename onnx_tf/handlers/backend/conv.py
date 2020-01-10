from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from .conv_mixin import ConvMixin


@onnx_op("Conv")
class Conv(ConvMixin, BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.conv(node, kwargs["tensor_dict"])

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls.conv(node, kwargs["tensor_dict"])
