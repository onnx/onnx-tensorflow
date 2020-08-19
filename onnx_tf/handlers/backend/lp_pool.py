from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from .pool_mixin import PoolMixin


@onnx_op("LpPool")
class LpPool(PoolMixin, BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    return cls.pool(node, kwargs["tensor_dict"], "LP",
                    kwargs.get("strict", True))

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_2(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)
