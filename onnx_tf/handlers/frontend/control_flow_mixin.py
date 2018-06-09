from .broadcast_mixin import BroadcastMixin


class LogicalMixin(BroadcastMixin):

  @classmethod
  def logical_op(cls, node, **kwargs):
    if cls.SINCE_VERSION <= 6:
      return cls.limited_broadcast(node, **kwargs)
    else:  # since_version >= 7
      return cls.np_broadcast(node, **kwargs)


class ComparisonMixin(BroadcastMixin):

  @classmethod
  def comparison_op(cls, node, **kwargs):
    if cls.SINCE_VERSION <= 6:
      return cls.limited_broadcast(node, **kwargs)
    else:  # since_version >= 7
      return cls.np_broadcast(node, **kwargs)
