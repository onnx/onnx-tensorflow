from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.frontend_handler import version
from .pool_mixin import PoolMixin


class MaxPool(PoolMixin, FrontendHandler):

  @classmethod
  @version(1)
  def version_1(cls, node, **kwargs):
    return cls.pool_op(node, **kwargs)
