import tensorflow as tf
import numpy as np

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func

from onnx_tf.common import exception
from onnx_tf.common import data_type
from onnx_tf.common import sys_config

tf.config.run_functions_eagerly(True)


@onnx_op("Unique")
@tf_func(tf.unique_with_counts)
@tf_func(tf.sort)
class Unique(BackendHandler):
  x_supported_types = [
      tf.uint8, tf.uint16, tf.uint32, tf.uint64, tf.int8, tf.int16, tf.int32,
      tf.int64, tf.float16, tf.double, tf.string, tf.bool, tf.complex64,
      tf.complex128
  ]
  axis_supported_types = [int]
  sorted_supported_types = axis_supported_types
  x_cast_map = {tf.uint32: tf.int64, tf.bool: None, tf.string: None}
  axis_case_map = {tf.uint8: tf.int32, tf.uint16: tf.int32}
  sorted_case_map = axis_case_map

  @classmethod
  def args_check(cls, node, **kwargs):
    cls.x_cast_map[tf.uint8] = tf.int8 if sys_config.auto_cast else None
    cls.x_cast_map[tf.uint16] = tf.float16 if sys_config.auto_cast else None
    cls.x_cast_map[tf.uint32] = tf.float32 if sys_config.auto_cast else None
    cls.x_cast_map[tf.uint64] = tf.float64 if sys_config.auto_cast else None
    cls.x_cast_map[tf.complex64] = tf.float64 if sys_config.auto_cast else None
    cls.x_cast_map[
        tf.complex128] = tf.float128 if sys_config.auto_cast else None
    cls.x_cast_map[tf.uint64] = tf.float64 if sys_config.auto_cast else None

    # Input:
    # data type is acceptable
    x = kwargs["tensor_dict"][node.inputs[0]]
    if tf.is_tensor(x):
      # x_shape = x.get_shape().as_list()
      # if len(x_shape) <= 0:  # meaning less
      #     exception.OP_UNSUPPORTED_EXCEPT("Unique required N-D input", "Tensorflow")
      x_dtype = x.dtype
      if x.dtype in cls.x_cast_map and cls.x_cast_map[
          x_dtype] is None:  # why is and
        exception.DTYPE_NOT_CAST_EXCEPT(
            "Unique input " + node.inputs[0] + " with data type '" +
            data_type.tf_to_np_str(x.dtype) + "'",
            data_type.tf_to_np_str_list(cls.x_supported_types))
    else:
      exception.OP_UNSUPPORTED_EXCEPT("Unique required N-D input", "Tensorflow")
    # Attributes:
    # axis: Optional, int, range is [-r, r - 1] where r = rank(input), default None.
    unique_axis = node.attrs.get("axis", -1)
    # print("unique_axis :{0}".format(unique_axis))
    if tf.is_tensor(unique_axis):
      axis_type = unique_axis.dtype
    else:
      axis_type = type(unique_axis)
    if not (axis_type in cls.axis_supported_types or axis_type is None):
      exception.DTYPE_NOT_CAST_EXCEPT(
          "Unique axis " + unique_axis + " with data type '" +
          data_type.tf_to_np_str(axis_type) + "'",
          data_type.tf_to_np_str_list(cls.axis_supported_types))
    rank_x = tf.rank(x).numpy()
    if unique_axis >= rank_x or unique_axis < (0 - rank_x):
      exception.OP_UNSUPPORTED_EXCEPT(
          "Unique required axis: None or in rand [-r, r - 1] where r = rank(input)."
      )
    # Attributes:
    # sorted: Optional, int, (0 or 1), default is 1.
    if_sorted = node.attrs.get("sorted", 1)
    if tf.is_tensor(if_sorted):
      if_sorted_type = if_sorted.dtype
    else:
      if_sorted_type = type(if_sorted)
    if not (if_sorted_type in cls.sorted_supported_types or
            if_sorted_type is None):
      exception.DTYPE_NOT_CAST_EXCEPT(
          "Unique sort " + if_sorted + " with data type '" +
          data_type.tf_to_np_str(if_sorted_type) + "'",
          data_type.tf_to_np_str_list(cls.sorted_supported_types))
    if if_sorted != 0 and if_sorted != 1:
      exception.OP_UNSUPPORTED_EXCEPT(
          "Unique required sort: None, either 0 or 1.")

    return 0

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
      if 0 > axis > (0 - len(x_sorted.shape)):
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
