import tensorflow as tf

from onnx_tf.common import exception
from onnx_tf.common import data_type
from onnx_tf.common import sys_config
from onnx_tf.common.tf_helper import tf_shape
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from .gather_and_scatter_mixin import GatherAndScatterMixin


@onnx_op("ScatterElements")
class ScatterElements(GatherAndScatterMixin, BackendHandler):
  supported_types = [
      tf.uint8, tf.uint16, tf.uint32, tf.uint64, tf.int8, tf.int16, tf.int32,
      tf.int64, tf.bfloat16, tf.float16, tf.float32, tf.float64
  ]
  cast_map = {}

  @classmethod
  def args_check(cls, node, **kwargs):
    # update cast_map base on auto_cast flag
    cls.cast_map[tf.complex64] = tf.float64 if sys_config.auto_cast else None
    cls.cast_map[tf.complex128] = tf.float64 if sys_config.auto_cast else None

    data = kwargs["tensor_dict"][node.inputs[0]]
    data_dtype = data.dtype
    if data_dtype in cls.cast_map and cls.cast_map[data_dtype] is None:
      exception.DTYPE_NOT_CAST_EXCEPT(
          "ScatterElements input " + node.inputs[0] + " and " + node.inputs[2] +
          " with data type '" + data_type.tf_to_np_str(data_dtype) + "'",
          data_type.tf_to_np_str_list(cls.supported_types))

  @classmethod
  def _common(cls, node, **kwargs):
    axis = node.attrs.get("axis", 0)
    data = kwargs["tensor_dict"][node.inputs[0]]
    indices = kwargs["tensor_dict"][node.inputs[1]]
    updates = kwargs["tensor_dict"][node.inputs[2]]
    data_dtype = data.dtype

    # poocess negative axis
    axis = axis if axis >= 0 else tf.add(tf.rank(data), axis)

    # check are there any indices are out of bounds
    result = cls.chk_idx_out_of_bounds_along_axis(data, axis, indices)
    msg = 'ScatterElements indices are out of bounds, please double check the indices and retry.'
    with tf.control_dependencies(
        [tf.compat.v1.assert_equal(result, True, message=msg)]):
      # process negative indices
      indices = cls.process_neg_idx_along_axis(data, axis, indices)

      # Calculate shape of the tensorflow version of indices tensor.
      sparsified_dense_idx_shape = tf_shape(updates)

      # Move on to convert ONNX indices to tensorflow indices in 2 steps:
      #
      # Step 1:
      #   What would the index tensors look like if updates are all
      #   dense? In other words, produce a coordinate tensor for updates:
      #
      #   coordinate[i, j, k ...] = [i, j, k ...]
      #   where the shape of "coordinate" tensor is same as that of updates.
      #
      # Step 2:
      #   But the coordinate tensor needs some correction because coord
      #   vector at position axis is wrong (since we assumed update is dense,
      #   but it is not at the axis specified).
      #   So we update coordinate vector tensor elements at psotion=axis with
      #   the sparse coordinate indices.

      idx_tensors_per_axis = [
          tf.range(sparsified_dense_idx_shape[i])
          for i in range(updates.shape.rank)
      ]
      idx_tensors_per_axis = tf.meshgrid(*idx_tensors_per_axis, indexing='ij')
      idx_tensors_per_axis[axis] = indices
      dim_expanded_idx_tensors_per_axis = [
          tf.expand_dims(idx_tensor, axis=-1)
          for idx_tensor in idx_tensors_per_axis
      ]
      coordinate = tf.concat(dim_expanded_idx_tensors_per_axis, axis=-1)

      # Now the coordinate tensor is in the shape
      # [updates.shape, updates.rank]
      # we need it to flattened into the shape:
      # [product(updates.shape), updates.rank]
      indices = tf.reshape(coordinate, [-1, tf.rank(data)])
      updates = tf.reshape(updates, [-1])

      # process tf.tensor_scatter_nd_update unsupported datatype for data and updates
      data = tf.cast(
          data,
          cls.cast_map[data_dtype]) if data_dtype in cls.cast_map else data
      updates = tf.cast(
          updates,
          cls.cast_map[data_dtype]) if data_dtype in cls.cast_map else updates
      output = tf.tensor_scatter_nd_update(data, indices, updates)
      return [
          tf.cast(output, data_dtype) if data_dtype in cls.cast_map else output
      ]

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
