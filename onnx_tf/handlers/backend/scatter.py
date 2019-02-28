import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Scatter")
@tf_func(tf.scatter_update)
class Scatter(BackendHandler):

  @classmethod
  def version_9(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    axis = node.attrs.get("axis", 0)
    assert axis == 0, "Non-zero axis is not supported in Scatter."

    data = tensor_dict[node.inputs[0]]
    indices = tensor_dict[node.inputs[1]]
    updates = tensor_dict[node.inputs[2]]

    data_shape = tf.cast(tf.shape(data), indices.dtype)
    dense_update = tf.scatter_nd(indices, updates, data_shape)
    update_mask = tf.scatter_nd(indices, tf.ones_like(updates), data_shape)
    output_data = data * (tf.cast(1.0, dtype=update_mask.dtype) - update_mask) + dense_update

    return [output_data]