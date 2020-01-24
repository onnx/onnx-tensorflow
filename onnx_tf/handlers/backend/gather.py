import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Gather")
@tf_func(tf.gather)
class Gather(BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]

  @classmethod
  def version_11(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    indices = kwargs["tensor_dict"][node.inputs[1]]
    axis = node.attrs.get("axis", 0)
    dimension = tf.cast(tf.shape(x)[axis], indices.dtype)
    indices_shape = tf.shape(indices)
    dim_broadcasted = tf.broadcast_to(dimension, indices_shape)
    nonneg_indices = tf.cast(tf.greater_equal(indices, 0), dtype=indices.dtype)
    negative_indices = tf.cast(tf.less(indices, 0), dtype=indices.dtype)
    final_indices = dim_broadcasted * negative_indices + negative_indices * indices + nonneg_indices * indices
    cond1 = tf.greater_equal(final_indices, 0)
    cond2 = tf.less(final_indices, dimension)
    indice_valiation = tf.reduce_all(tf.logical_and(cond1, cond2))
    msg = 'Gather indices are out of bounds, please double check the indices and retry.'
    with tf.control_dependencies(
        [tf.compat.v1.assert_equal(indice_valiation, True, message=msg)]):
      return [tf.gather(x, final_indices, axis=axis, name=cls.tf_node_name(node.name))]
