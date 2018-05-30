from onnx_tf.common.broadcast import get_broadcast_axis


class LogicalMixin(object):

  @classmethod
  def logical_op(cls, node, **kwargs):
    if cls.SINCE_VERSION <= 6:
      return cls._limited_broadcast(node, **kwargs)
    else:  # since_version >= 7
      return cls._np_broadcast(node, **kwargs)

  @classmethod
  def _limited_broadcast(cls, node, broadcast=1, **kwargs):
    ex_kwargs = {}
    if broadcast == 1:
      ex_kwargs["broadcast"] = 1
      node_dict = kwargs["node_dict"]
      axis = get_broadcast_axis(*[node_dict[x] for x in node.inputs[0:2]])
      if axis is not None:
        ex_kwargs["axis"] = axis
    return cls.make_node_from_tf_node(node, **ex_kwargs)

  @classmethod
  def _np_broadcast(cls, node, **kwargs):
    return cls.make_node_from_tf_node(node)


class ComparisonMixin(object):

  @classmethod
  def comparison_op(cls, node, **kwargs):
    if cls.SINCE_VERSION <= 6:
      return cls._limited_broadcast(node, **kwargs)
    else:  # since_version >= 7
      return cls._np_broadcast(node, **kwargs)

  @classmethod
  def _limited_broadcast(cls, node, **kwargs):
    ex_kwargs = {"broadcast": 1}
    node_dict = kwargs["node_dict"]
    axis = get_broadcast_axis(*[node_dict[x] for x in node.inputs[0:2]])
    if axis is not None:
      ex_kwargs["axis"] = axis
    return cls.make_node_from_tf_node(node, **ex_kwargs)

  @classmethod
  def _np_broadcast(cls, node, **kwargs):
    return cls.make_node_from_tf_node(node)
