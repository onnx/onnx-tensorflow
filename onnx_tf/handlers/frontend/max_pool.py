from onnx_tf.handlers.frontend_handler import FrontendHandler
from .pool_mixin import PoolMixin


class MaxPool(PoolMixin, FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.pool_op(node, **kwargs)
