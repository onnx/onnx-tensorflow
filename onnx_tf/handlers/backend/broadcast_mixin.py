import numpy as np
import tensorflow as tf


def is_tensor_or_var(obj):
  return isinstance(obj, tf.Tensor) or isinstance(obj, tf.Variable)


class BroadcastMixin(object):

  @classmethod
  def explicit_broadcast(cls, inputs, axis=None, tensor_dict=None):
    x = inputs[0] if is_tensor_or_var(inputs[0]) else tensor_dict[inputs[0]]
    y = inputs[1] if is_tensor_or_var(inputs[1]) else tensor_dict[inputs[1]]

    if np.prod(y.shape) == 1:
      return y

    if not is_tensor_or_var(x) or not is_tensor_or_var(y):
      raise ValueError("Targets for explicit broadcasting need to be Tensor.")

    if axis is None:
      return y

    total_num_dim = len(x.get_shape())
    if axis < 0:
      axis += total_num_dim

    if axis + len(y.get_shape()) == total_num_dim:
      return y

    dims = [axis + i for i in range(len(y.get_shape()))]
    new_y = y
    for i in range(total_num_dim):
      if i not in dims:
        new_y = tf.expand_dims(new_y, i)
    return new_y

  @classmethod
  def limited_broadcast(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]
    y = tensor_dict[node.inputs[1]]
    if node.attrs.get("broadcast") == 1:
      y = cls.explicit_broadcast([x, y], node.attrs.get("axis", None))
      return [cls.make_tensor_from_onnx_node(node, inputs=[x, y], **kwargs)]
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]
