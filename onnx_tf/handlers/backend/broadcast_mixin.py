import tensorflow as tf


class BroadcastMixin(object):

  @classmethod
  def explicit_broadcast(cls, inputs, axis, tensor_dict):
    x = tensor_dict[inputs[0]]
    y = tensor_dict[inputs[1]]

    if axis is None:
      return [x, y]

    total_num_dim = len(x.get_shape())
    if axis < 0:
      axis += total_num_dim

    if axis + len(y.get_shape()) == total_num_dim:
      return [x, y]

    dims = [axis + i for i in range(len(y.get_shape()))]
    for i in range(total_num_dim):
      if i not in dims:
        new_y = tf.expand_dims(y, i)
    return new_y

  @classmethod
  def limited_broadcast(cls, node, **kwargs):
    if node.attrs.get("broadcast") == 1:
      y = cls.explicit_broadcast(node.inputs, node.attrs.get("axis", None),
                                 kwargs["tensor_dict"])
      return [cls.make_tf_tensor(node, inputs=[node.inputs[0], y], **kwargs)]
    return [cls.make_tf_tensor(node, **kwargs)]
