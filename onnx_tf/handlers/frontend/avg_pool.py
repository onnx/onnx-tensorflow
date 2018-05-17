from .pool_common import PoolCommon


class AvgPool(PoolCommon):
  _ONNX_OP = "AveragePool"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.pool_op(node, 1, **kwargs)
