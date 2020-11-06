import tensorflow as tf

from onnx_tf.common.tf_helper import tf_shape


class UnpoolMixin(object):

  @classmethod
  @tf.autograph.experimental.do_not_convert()
  def max_unpool(cls, node, input_dict):
    """
            MaxUnpooling operation
    """
    x = input_dict[node.inputs[0]]
    ind = input_dict[node.inputs[1]]
    if len(node.inputs) > 2:
      output_shape = input_dict.get(node.inputs[2], None)
    else:
      output_shape = None

    kernel_shape = node.attrs["kernel_shape"]

    spatial_size = len(kernel_shape)

    # if strides are not provided default is 1 along each spatial axis
    strides = node.attrs.get("strides", [1] * spatial_size)
    pads = node.attrs.get("pads", None)

    input_shape = tf_shape(x)
    default_shape = cls._get_default_shape(input_shape, kernel_shape, strides)
    default_shape = [input_shape[0]] + [input_shape[1]] + default_shape

    unpooled = cls._unpool(x, ind, default_shape)

    if output_shape is not None:
      pads = cls._get_pads_from_output_shape(unpooled, output_shape)
    if pads is not None:
      unpooled = cls._pad_output(unpooled, pads, 0)

    return [unpooled]

  @classmethod
  def _get_default_shape(cls, input_shape, kernel_shape, strides):
    """
            Calculates default shape from kernel_shape and strides
            Args:
                input_shape:   shape of the input to unpool op
                kernel_shape:  the size of the kernel along each axis
                output_shape:  stride along each spatial axis
          Return:
            default_shape: calculated default_shape
    """
    default_shape = []
    for d in range(len(kernel_shape)):
      default_shape.append((input_shape[d + 2] - 1) * int(strides[d]) +
                           int(kernel_shape[d]))
    return default_shape

  @classmethod
  def _get_pads_from_output_shape(cls, unpool, output_shape):
    """
            Calculates the paddings from specified output_shape
            Args:
                unpool:       result from unpool operation
                output_shape: expected shape of the output
            Return:
                pads:         calculated paddings in format
                              [x1_begin, x2_begin,.., x1_end, x2_end]
                              where xi_... represent pads added to begin
                              or end of axis i
    """
    unpool_shape = tf.cast(tf.shape(unpool), dtype=tf.int32)
    new_shape = tf.cast(output_shape, dtype=tf.int32)

    pads_begin = []
    pads_end = []

    for d in range(len(unpool.get_shape())):
      pad_total = new_shape[d] - unpool_shape[d]
      pad_begin = tf.cast(pad_total / 2, tf.int32)
      pad_end = pad_total - pad_begin
      pads_begin = pads_begin + [pad_begin]
      pads_end = pads_end + [pad_end]

    pads = pads_begin + pads_end
    return pads

  @classmethod
  def _pad_output(cls, unpool, pads, constant_values):
    """
            Pad the output from unpool op
            Args:
                unpool:         result from unpool op
                pads:           paddings in format
                                [x1_begin, x2_begin,..., x1_end, x2_end]
                constant_values: constant value to fill up the padded spaces
            Return:
                padded:         padded tensor
    """
    unpool_shape = unpool.get_shape()
    paddings = []
    for d in range(len(unpool_shape)):
      paddings = paddings + [[pads[d], pads[d + len(unpool_shape)]]]
    padded = tf.pad(unpool,
                    paddings,
                    'CONSTANT',
                    constant_values=constant_values)
    return padded

  @classmethod
  def _unpool(cls, pool, ind, output_shape, scope='unpool'):
    """
            Unpooling layer after max_pool_with_argmax.

            Args:
                pool:          max pooled output tensor
                ind:           argmax indices
                output_shape:  the shape of the output
            Return:
                unpool:        unpooling tensor
    """
    with tf.compat.v1.variable_scope(scope):
      input_shape = tf.shape(pool)

      flat_input_size = tf.reduce_prod(input_shape)
      flat_output_shape = [tf.reduce_prod(output_shape)]

      pool_ = tf.reshape(pool, [flat_input_size])
      ind_ = tf.reshape(ind, [flat_input_size, 1])

      ret = tf.scatter_nd(ind_,
                          pool_,
                          shape=tf.cast(flat_output_shape, tf.int64))
      ret = tf.reshape(ret, output_shape)
    return ret
