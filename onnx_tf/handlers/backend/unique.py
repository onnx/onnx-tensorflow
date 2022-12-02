import tensorflow as tf
import numpy as np

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.common import exception

tf.config.run_functions_eagerly(True)


@onnx_op("Unique")
class Unique(BackendHandler):

  @classmethod
  def version_11(cls, node, **kwargs):
    # Get input x
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]
    # Get attribution axis
    axis = node.attrs.get("axis", 0)
    # Get attribution sorted
    if_sorted = node.attrs.get("sorted", 1)

    # Step 1: Action to sort
    if if_sorted == 1:
      x_sorted = tf.sort(x, axis, 'ASCENDING')
    else:
      x_sorted = x

    # Step 2: Action to unique
    if len(x_sorted.shape) == 1:
      x_picked = x_sorted
      y, inverse_idx, counts = tf.unique_with_counts(x_picked, out_idx=tf.int64)
      y_target = tf.convert_to_tensor(y)
      indices = []
      for item in y.numpy().tolist():
        indices.append(np.argmax(x == item))
      inverse_indices = []
      if if_sorted == 1:
        for item in x.numpy().tolist():
          inverse_indices.append(np.argmax(y == item))
      else:
        inverse_indices = inverse_idx
    else:
      if axis >= len(x_sorted.shape) or axis < (0 - len(x_sorted.shape)):
        exception.OP_UNSUPPORTED_EXCEPT(
            "Unique required axis: None or in rand [-r, r - 1] where r = rank(input)."
        )
      elif 0 > axis > (0 - len(x_sorted.shape)):
        axis += len(x_sorted.shape)
      # split X based on axis
      idx = tf.reshape(np.arange(x_sorted.shape[axis]), -1)
      if axis > 0:
        x_picked = tf.gather(x_sorted, idx, axis=axis)[0]
        x_compare_base = tf.gather(x, idx, axis=axis)[0]
      elif axis == 0:
        x_picked = tf.gather(x_sorted, idx, axis=axis)
        x_compare_base = x
      # Construction Y, indices and counts
      y_target = []
      indices = []
      counts = []
      for item in x_picked.numpy().tolist():
        if item not in y_target:
          y_target.append(item)
          item_first_location_in_x = tf.argmax(x_compare_base == item)
          if tf.rank(item_first_location_in_x).numpy(
          ) == 0:  # use tf.rank instead of len(shape) for int
            indices.append(item_first_location_in_x.numpy().tolist())
          else:
            indices.append(item_first_location_in_x.numpy().tolist()[0])
          counts.append(1)
        else:
          # exist item, skip updating indices
          item_location_in_y = np.argmax(y_target == item)
          counts[item_location_in_y] += 1
      # Construction inverse_indices
      inverse_indices = []
      for item in x_compare_base.numpy().tolist():
        item_location_in_y = tf.argmax(
            tf.convert_to_tensor(y_target) == tf.convert_to_tensor(item))
        if tf.rank(item_location_in_y).numpy() == 0:
          item_location_in_y = item_location_in_y.numpy().tolist()
        else:
          item_location_in_y = item_location_in_y.numpy().tolist()[0]
        inverse_indices.append(item_location_in_y)
      # Complete Y
      if axis > 0:
        y_target = tf.gather(x, indices, axis=axis)

    return tf.convert_to_tensor(y_target), tf.convert_to_tensor(indices, dtype=tf.int64), \
           tf.convert_to_tensor(inverse_indices, dtype=tf.int64), tf.convert_to_tensor(counts, dtype=tf.int64)
