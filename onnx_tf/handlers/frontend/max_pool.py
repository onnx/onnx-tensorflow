from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op
from .pool_mixin import PoolMixin


@onnx_op("MaxPool")
@tf_op("MaxPool")
class MaxPool(PoolMixin, FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.pool_op(node, **kwargs)

  @classmethod
  def version_8(cls, node, **kwargs):
    return cls.pool_op(node, **kwargs)
