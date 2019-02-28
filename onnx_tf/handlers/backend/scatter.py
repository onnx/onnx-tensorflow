import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Scatter")
@tf_func(tf.scatter_nd)
class Scatter(BackendHandler):

  @classmethod
  def version_9(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    axis = node.attrs.get("axis", 0)
    assert axis == 0, "Non-zero axis is not supported in Scatter."

    # ONNX scatter is really scatter update in tensorflow.
    # We do so in 3 steps:
    # 1. Create dense update tensor (tensor with updates scattered and
    #       zero everywhere else) U.
    # 2. Create dense update mask (tensor with the same shape as U,
    #       with 1 at non-zero positions in U and 0 everywhere else) M.
    # 3. The ONNX output tensor is therefore O = data * (1-M) + U.
    data = tensor_dict[node.inputs[0]]
    indices = tensor_dict[node.inputs[1]]
    updates = tensor_dict[node.inputs[2]]

    data_shape = tf.cast(tf.shape(data), indices.dtype)
    dense_update = tf.scatter_nd(indices, updates, data_shape)
    update_mask = tf.scatter_nd(indices, tf.ones_like(updates), data_shape)
    output_data = data * (
        tf.cast(1.0, dtype=update_mask.dtype) - update_mask) + dense_update

    return [output_data]
