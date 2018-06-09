class BroadcastMixin(object):

  # Until version 6
  @staticmethod
  def get_broadcast_axis(x, y):
    # TODO(fumihwh): if works with input from prev-node has multiple outputs?
    x_shape = x.attr["_output_shapes"][0]
    y_shape = y.attr["_output_shapes"][0]
    y_dim = len(y_shape)
    if x_shape == y_shape:
      return None
    for i in range(len(x_shape)):
      if x_shape[i:i + y_dim] == y_shape:
        return i

  @classmethod
  def limited_broadcast(cls, node, broadcast=1, **kwargs):
    ex_kwargs = {}
    if broadcast == 1:
      ex_kwargs["broadcast"] = 1
      node_dict = kwargs["node_dict"]
      axis = cls.get_broadcast_axis(*[node_dict[x] for x in node.inputs[0:2]])
      if axis is not None:
        ex_kwargs["axis"] = axis
    return cls.make_node_from_tf_node(node, **ex_kwargs)

  @classmethod
  def np_broadcast(cls, node, **kwargs):
    return cls.make_node_from_tf_node(node)
