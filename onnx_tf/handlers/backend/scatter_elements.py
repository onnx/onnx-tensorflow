import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("ScatterElements")
class ScatterElements(BackendHandler):

  @classmethod
  def version_11(cls, node, **kwargs):
    axis = node.attrs.get("axis", 0)
    data = kwargs["tensor_dict"][node.inputs[0]]
    indices = kwargs["tensor_dict"][node.inputs[1]]
    updates = kwargs["tensor_dict"][node.inputs[2]]

    # Calculate shape of the tensorflow version of indices tensor.
    sparsified_dense_idx_shape = updates.get_shape().as_list()

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

    idx_tensors_per_axis = tf.meshgrid(*list(
        map(lambda x: tf.range(x, dtype=tf.dtypes.int64),
            sparsified_dense_idx_shape)),
                                       indexing='ij')
    idx_tensors_per_axis[axis] = indices
    dim_expanded_idx_tensors_per_axis = list(
        map(lambda x: tf.expand_dims(x, axis=-1), idx_tensors_per_axis))
    coordinate = tf.concat(dim_expanded_idx_tensors_per_axis, axis=-1)

    # Now the coordinate tensor is in the shape
    # [updates.shape, updates.rank]
    # we need it to flattened into the shape:
    # [product(updates.shape), updates.rank]
    indices = tf.reshape(coordinate, [-1, tf.rank(data)])
    updates = tf.reshape(updates, [-1])

    return [tf.tensor_scatter_nd_update(data, indices, updates)]
