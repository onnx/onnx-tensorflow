from .pool_common import PoolCommon


class MaxPool(PoolCommon):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.pool_op(node, **kwargs)
