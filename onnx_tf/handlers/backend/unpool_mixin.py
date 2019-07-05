import tensorflow as tf

from onnx_tf.common import get_data_format
from onnx_tf.common import get_perm_from_formats

class UnpoolMixin(object):

  @classmethod
  def max_unpool(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    ind = input_dict[node.inputs[1]]

    x_rank = len(x.get_shape())
    storage_format, compute_format = get_data_format(x_rank)
    spatial_size = x_rank - 2

    kernel_shape = node.attrs["kernel_shape"]
    # if strides are not provided default is same as the kernel
    strides = node.attrs.get("strides", kernel_shape)
    pads = node.attrs.get("pads", [0] * spatial_size)
    output_shape = node.attrs.get("output_shape", None)

    input_shape = x.get_shape()
    # if output_shape is not provided, calculate it
    if output_shape is None:
      output_shape = []
      for d in range(len(kernel_shape)):
        output_shape.append((int(input_shape[d + 2]) - 1) * int(strides[d]) +
                            int(kernel_shape[d]) - 2 * int(pads[d]))
    output_shape = [int(input_shape[0])] + output_shape + [int(input_shape[1])]

    need_trans = storage_format != "NHWC"
    if need_trans:
      x = tf.transpose(x, perm=get_perm_from_formats(storage_format, "NHWC"))
      ind = tf.transpose(ind, perm=get_perm_from_formats(storage_format, "NHWC"))

    unpooled = cls.unpool(x, ind, output_shape)

    if need_trans:
      unpooled = tf.transpose(
        unpooled, perm=get_perm_from_formats("NHWC", storage_format))

    return [unpooled]

  @classmethod
  def unpool(cls, pool, ind, output_shape, scope='unpool'):
    """
     Unpooling layer after max_pool_with_argmax.

     Args:
         pool:          max pooled output tensor
         ind:           argmax indices
         output_shape:  the shape of the output
     Return:
         unpool:        unpooling tensor
    """
    with tf.variable_scope(scope):
      input_shape = tf.shape(pool)

      flat_input_size = tf.reduce_prod(input_shape)
      flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

      pool_ = tf.reshape(pool, [flat_input_size])
      batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                          shape=[input_shape[0], 1, 1, 1])
      b = tf.ones_like(ind) * batch_range
      b1 = tf.reshape(b, [flat_input_size, 1])
      ind_ = tf.reshape(ind, [flat_input_size, 1])
      ind_ = tf.concat([b1, ind_], 1)

      ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
      ret = tf.reshape(ret, output_shape)

      set_input_shape = pool.get_shape()
      ret.set_shape(output_shape)
      return ret
