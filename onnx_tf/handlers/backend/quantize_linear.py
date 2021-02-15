import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.common.tf_helper import tf_shape


@onnx_op("QuantizeLinear")
class QuantizeLinear(BackendHandler):

  @classmethod
  def version_10(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]
    y_scale = tensor_dict[node.inputs[1]]

    x = tf.cast(x, tf.float32)
    y = tf.divide(x, y_scale)
    y = tf.round(y)
    if len(node.inputs) == 3:
      y_zero_point = tensor_dict[node.inputs[2]]
      y_dtype = y_zero_point.dtype
      y_zero_point = tf.cast(y_zero_point, tf.float32)
      y = tf.add(y, y_zero_point)
    else:  # y_zero_point default dtype = uint8
      y_dtype = tf.uint8

    y = tf.saturate_cast(y, y_dtype)

    return [y]

  @classmethod
  def version_13(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]
    y_scale = tensor_dict[node.inputs[1]]
    axis = node.attrs.get("axis", 1)

    x = tf.cast(x, tf.float32)
    x_shape = tf_shape(x)
    x_rank = len(x_shape)
    y_scale_shape = tf_shape(y_scale)
    y_scale_rank = len(y_scale_shape)

    # Reshape process is needed for per-axis quantization
    # when scale is a 1-D tensor
    if y_scale_rank == 1:
      shape_broadcast = list([1 for _ in range(axis)] + [x_shape[axis]] +
                             [1 for _ in range(axis + 1, x_rank)])
      y_scale = tf.reshape(y_scale, shape_broadcast)

    y = tf.divide(x, y_scale)
    y = tf.round(y)
    if len(node.inputs) == 3:
      y_zero_point = tensor_dict[node.inputs[2]]
      y_dtype = y_zero_point.dtype
      y_zero_point = tf.cast(y_zero_point, tf.float32)
      y_zero_point = tf.reshape(
          y_zero_point, shape_broadcast) if y_scale_rank == 1 else y_zero_point
      y = tf.add(y, y_zero_point)
    else:  # y_zero_point default dtype = uint8
      y_dtype = tf.uint8

    y = tf.saturate_cast(y, y_dtype)

    return [y]
