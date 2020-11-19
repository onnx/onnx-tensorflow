import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from .gather_and_scatter_mixin import GatherAndScatterMixin


@onnx_op("GatherElements")
class GatherElements(GatherAndScatterMixin, BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    # GatherElements takes two inputs data and indices of the same rank r >= 1 and an optional attribute axis that identifies
    # an axis of data (by default, the outer-most axis, that is axis 0). It is an indexing operation that produces its output by
    # indexing into the input data tensor at index positions determined by elements of the indices tensor. Its output shape is the
    # same as the shape of indices and consists of one value (gathered from the data) for each element in indices.

    axis = node.attrs.get("axis", 0)
    data = kwargs["tensor_dict"][node.inputs[0]]
    indices = kwargs["tensor_dict"][node.inputs[1]]

    # poocess negative axis
    axis = axis if axis >= 0 else tf.add(tf.rank(data), axis)

    # check are there any indices are out of bounds
    result = cls.chk_idx_out_of_bounds_along_axis(data, axis, indices)
    msg = 'GatherElements indices are out of bounds,'\
      ' please double check the indices and retry.'
    with tf.control_dependencies(
        [tf.compat.v1.assert_equal(result, True, message=msg)]):
      # process negative indices
      indices = cls.process_neg_idx_along_axis(data, axis, indices)

      # adapted from reference implementation in onnx/onnx/backend/test/case/node/gatherelements.py
      if axis == 0:
        axis_perm = tf.range(tf.rank(data))
        data_swaped = data
        index_swaped = indices
      else:
        axis_perm = tf.tensor_scatter_nd_update(tf.range(tf.rank(data)),
                                                tf.constant([[0], [axis]]),
                                                tf.constant([axis, 0]))
        data_swaped = tf.transpose(data, perm=axis_perm)
        index_swaped = tf.transpose(indices, perm=axis_perm)

      idx_tensors_per_axis = [
          tf.range(tf.shape(index_swaped, index_swaped.dtype)[i])
          for i in range(index_swaped.shape.rank)
      ]
      idx_tensors_per_axis = tf.meshgrid(*idx_tensors_per_axis, indexing='ij')
      idx_tensors_per_axis[0] = index_swaped
      dim_expanded_idx_tensors_per_axis = [
          tf.expand_dims(idx_tensor, axis=-1)
          for idx_tensor in idx_tensors_per_axis
      ]
      index_expanded = tf.concat(dim_expanded_idx_tensors_per_axis, axis=-1)

      gathered = tf.gather_nd(data_swaped, index_expanded)
      y = tf.transpose(gathered, perm=axis_perm)

      return [y]

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
