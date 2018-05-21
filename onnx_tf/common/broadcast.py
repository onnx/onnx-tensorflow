import tensorflow as tf


def explicit_broadcast(tensor, broadcast_dim):
  total_num_dim = len(tensor.get_shape().as_list())
  if broadcast_dim < 0:
    broadcast_dim += total_num_dim
  dims = [broadcast_dim + i for i in range(len(tensor.shape))]
  for i in range(total_num_dim):
    if i not in dims:
      tensor = tf.expand_dims(tensor, i)


# Until version 6
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
