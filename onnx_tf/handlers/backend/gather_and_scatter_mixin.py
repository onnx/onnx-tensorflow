import tensorflow as tf

from onnx_tf.common.tf_helper import tf_shape


class GatherAndScatterMixin(object):

  @classmethod
  def chk_idx_out_of_bounds(cls, data, indices, batch_dims=0):
    """ Check indices out of bounds for ScatterND and GatherND
    In Tensorflow GPU version, if an out of bound index is found,
    a 0 is stored in the corresponding output value for GatherND;
    and the index is ignored for ScatterND/TensorScatterNDUpdate.
    But ONNX spec state that it is an error if any index values
    are out of bounds. Therefore the converter need to run this
    function to verify all the indices are in bounds before send
    it to Tensoflow. If out of bound is detected then the caller
    of this function need to throw InvalidArgumentError exception.
    """
    data_shape = tf_shape(data)
    indices_shape = tf_shape(indices)
    if batch_dims > 0:
      new_shape = indices_shape[0]
      for d in range(1, batch_dims):
        new_shape = tf.multiply(new_shape, indices_shape[d])
      new_shape = [new_shape, indices_shape[-1]]
      indices = tf.reshape(indices, new_shape)

    def _chk_idx_out_of_bounds(i, result):
      indices_i = tf.transpose(indices)[i]
      limit_i = tf.cast(data_shape, indices.dtype)[i + batch_dims]
      cond1 = tf.greater_equal(indices_i, tf.negative(limit_i))
      cond2 = tf.less(indices_i, limit_i)
      result = tf.reduce_all(tf.logical_and(cond1, cond2))
      return i + 1, result

    _, result = tf.while_loop(
        lambda i, result: tf.logical_and(tf.less(i, indices_shape[-1]), result),
        _chk_idx_out_of_bounds, [tf.zeros([], tf.int64), True])
    return result

  @classmethod
  def chk_idx_out_of_bounds_along_axis(cls, data, axis, indices):
    """ Check indices out of bounds for ScatterElement
    In Tensorflow GPU version, if an out of bound index is found,
    the index is ignored for ScatterND/TensorScatterNDUpdate.
    But ONNX spec state that it is an error if any index values
    are out of bounds. Therefore the converter need to run this
    function to verify all the indices are in bounds along the
    axis before send it to Tensoflow. If out of bound is detected
    then the caller of this function need to throw
    InvalidArgumentError exception.
    """
    data_shape = tf.cast(tf_shape(data), indices.dtype)
    limit = data_shape[axis]
    cond1 = tf.greater_equal(indices, tf.negative(limit))
    cond2 = tf.less(indices, limit)
    return tf.logical_and(cond1, cond2)

  @classmethod
  def process_neg_idx(cls, data, indices, batch_dims=0):
    """ Convert all the negative indices to positive
    GatherND and ScatterND/TensorScatterNDUpdate in Tensorflow
    doesn't support negative indices. Therefore need to run this
    function to convert all the negative indices to positive before
    send it to Tensorflow.
    """
    data_shape = tf_shape(data)
    if data.get_shape().is_fully_defined():
      indices_shape = indices.get_shape().as_list()
    else:
      indices_shape = tf_shape(indices)
    if batch_dims > 0:
      max_i = tf.cast(data_shape[batch_dims:indices_shape[-1] + batch_dims],
                      indices.dtype)
    else:
      max_i = tf.cast(data_shape[:indices_shape[-1]], indices.dtype)
    return tf.math.floormod(tf.add(indices, max_i), max_i)

  @classmethod
  def process_neg_idx_along_axis(cls, data, axis, indices):
    """ Convert all the negative indices to positive
    ScatterND/TensorScatterNDUpdate in Tensorflow doesn't support
    negative indices. Therefore need to run this function to convert
    all the negative indices to positive before send it to Tensorflow.
    """
    data_shape = tf_shape(data)
    max_i = tf.cast(data_shape[axis], indices.dtype)
    return tf.math.floormod(tf.add(indices, max_i), max_i)
