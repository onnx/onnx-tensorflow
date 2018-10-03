import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("DynamicSlice")
@tf_func(tf.slice)
class DynamicSlice(BackendHandler):

  @classmethod
  def version_9(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    input_tensor = tensor_dict[node.inputs[0]]
    starts = tensor_dict[node.inputs[1]]
    ends = tensor_dict[node.inputs[2]]
    l = list(range(starts.shape[0]))
    axes = tensor_dict[node.inputs[3]] if len(
        node.inputs) == 4 else tf.constant(l, ends.dtype)

    # first of all, compute sparse shape, that is:
    # for (axis in axes):
    #   sparse_shape[axis] = input_tensor.shape[axis]
    input_tensor_shape = tf.constant(input_tensor.shape.dims, ends.dtype)

    # expand a dimension of 1 at the end
    sparse_indices = tf.expand_dims(axes, -1)

    # build the indexed dimension sizes as sparse_shape
    sparse_shape = tf.gather_nd(
        params=input_tensor.shape, indices=sparse_indices)
    sparse_shape = tf.cast(sparse_shape, ends.dtype)

    # take care of starts, ends that are larger than the dim size.
    starts_min = tf.minimum(starts, sparse_shape)
    ends_min = tf.minimum(ends, sparse_shape)

    # take care of starts, ends that are negative
    is_starts_negative = tf.less(starts_min, tf.zeros_like(starts_min))
    starts_final = tf.where(is_starts_negative, starts_min + sparse_shape,
                            starts_min)
    is_ends_negative = tf.less(ends_min, tf.zeros_like(ends_min))
    ends_final = tf.where(is_ends_negative, ends_min + sparse_shape, ends_min)

    # need to densify everything for the inputs to slice
    # the output shape is the input_tensor rank
    output_shape = tf.reshape(tf.rank(input_tensor), [1])
    output_shape = tf.cast(output_shape, ends.dtype)

    # create dense tensor, pad 0 as default begins
    dense_begins = tf.sparse_to_dense(sparse_indices, output_shape,
                                      starts_final)
    # create dense tensor, pad -1 for next step
    dense_ends = tf.sparse_to_dense(
        sparse_indices,
        output_shape,
        ends_final,
        default_value=tf.constant(-1, dtype=dense_begins.dtype))
    # replace -1 with respective dimension sizes
    dense_ends = tf.where(
        tf.equal(dense_ends, tf.constant(-1, dtype=dense_begins.dtype)),
        input_tensor_shape, dense_ends)
    dense_sizes = dense_ends - dense_begins

    return [
        cls.make_tensor_from_onnx_node(
            node,
            inputs=[tensor_dict[node.inputs[0]], dense_begins, dense_sizes],
            **kwargs)
    ]
