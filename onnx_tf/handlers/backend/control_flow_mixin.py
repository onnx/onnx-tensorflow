from onnx_tf.common.broadcast import get_broadcast_axis


class LogicalMixin(object):

  @classmethod
  def process_attrs(cls, attrs):
    attrs.pop("axis", None)
    attrs.pop("broadcast", None)
    return attrs


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
