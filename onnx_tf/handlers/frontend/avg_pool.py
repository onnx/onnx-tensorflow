from onnx_tf.handlers.frontend_handler import FrontendHandler
from .pool_mixin import PoolMixin


class AvgPool(PoolMixin, FrontendHandler):
  ONNX_OP = "AveragePool"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.pool_op(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls.pool_op(node, **kwargs)
