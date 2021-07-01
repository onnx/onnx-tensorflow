import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Slice")
@tf_func(tf.strided_slice)
class Slice(BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]

    # Shape output as int64 since the spec implicitly allows int64
    full_sizes = tf.shape(x, out_type=tf.int64)

    starts = node.attrs.get("starts")
    ends = node.attrs.get("ends")
    slice_len = len(starts)
    axes = node.attrs.get("axes", list(range(slice_len)))

    updated_full_sizes = [0] * len(x.get_shape())
    updated_full_begin = [0] * len(x.get_shape())
    updated_starts = [0] * slice_len
    updated_ends = [0] * slice_len

    for axis in range(x.shape.rank):
      if axis not in axes:
        # Update the sizes for axes that are not in the axes attribute
        # No need to change the default of 0 in begins
        updated_full_sizes[axis] = full_sizes[axis]
      else:
        # Update the begins and sizes for each axis in the axes attribute
        for i in range(slice_len):
          if axis == axes[i]:
            updated_starts[i] = full_sizes[axis] + starts[i] if starts[
                i] < 0 else starts[i]
            updated_ends[
                i] = full_sizes[axis] + ends[i] if ends[i] < 0 else ends[i]
            if full_sizes[axis] is not None:
              updated_ends[i] = tf.reduce_min(
                  [full_sizes[axis], updated_ends[i]])
              updated_starts[i] = tf.reduce_min(
                  [full_sizes[axis], updated_starts[i]])

            updated_full_begin[axis] = updated_starts[i]
            updated_full_sizes[axis] = updated_ends[i] - updated_starts[i]

    return [
        cls.make_tensor_from_onnx_node(node,
                                       tf_func=tf.slice,
                                       inputs=[
                                           tensor_dict[node.inputs[0]],
                                           updated_full_begin,
                                           updated_full_sizes
                                       ],
                                       **kwargs)
    ]

  @classmethod
  def version_10(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    input_tensor = tensor_dict[node.inputs[0]]
    starts = tensor_dict[node.inputs[1]]
    ends = tensor_dict[node.inputs[2]]

    # first of all, get the input tensor shape
    input_tensor_shape = tf.shape(input_tensor, out_type=ends.dtype)

    axes = tensor_dict[node.inputs[3]] if len(node.inputs) >= 4 else tf.range(
        tf.shape(starts)[0], dtype=ends.dtype)

    is_axes_negative = tf.less(axes, tf.zeros_like(axes))
    axes = tf.where(is_axes_negative,
                    axes + tf.cast(tf.rank(input_tensor), axes.dtype), axes)

    # expand a dimension of 1 at the end
    sparse_indices = tf.cast(tf.expand_dims(axes, -1), tf.int64)

    # build the indexed dimension sizes as sparse_shape
    sparse_shape = tf.gather_nd(params=input_tensor_shape,
                                indices=sparse_indices)
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
    output_shape = tf.cast(output_shape, tf.int64)

    # create dense tensor, pad 0 as default begins
    dense_begins = tf.sparse.to_dense(
        tf.sparse.SparseTensor(sparse_indices, starts_final, output_shape))

    # create dense tensor, pad -1 for next step
    dense_ends = tf.sparse.SparseTensor(sparse_indices, ends_final,
                                        output_shape)
    dense_ends = tf.sparse.to_dense(dense_ends,
                                    default_value=tf.constant(
                                        -1, dtype=dense_begins.dtype))
    dense_ends = tf.where(
        tf.equal(dense_ends, tf.constant(-1, dtype=dense_begins.dtype)),
        input_tensor_shape, dense_ends)

    # create dense tensor for steps if not already so
    if len(node.inputs) >= 5:
      dense_steps = tf.sparse.SparseTensor(sparse_indices,
                                           tensor_dict[node.inputs[4]],
                                           output_shape)
      dense_steps = tf.sparse.to_dense(
          dense_steps,
          default_value=tf.constant(1, dtype=tensor_dict[node.inputs[4]].dtype))
    else:
      dense_steps = tf.ones(input_tensor_shape.shape, ends.dtype)

    return [
        cls.make_tensor_from_onnx_node(node,
                                       inputs=[
                                           tensor_dict[node.inputs[0]],
                                           dense_begins, dense_ends, dense_steps
                                       ],
                                       **kwargs)
    ]

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls.version_10(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls.version_10(node, **kwargs)
