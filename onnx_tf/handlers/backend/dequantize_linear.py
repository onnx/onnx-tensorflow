import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.common.tf_helper import tf_shape


@onnx_op("DequantizeLinear")
class DequantizeLinear(BackendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    if len(node.inputs) == 3:
      x = tensor_dict[node.inputs[0]]
      x_scale = tensor_dict[node.inputs[1]]
      x_zero_point = tensor_dict[node.inputs[2]]
      if x_scale.shape != x_zero_point.shape:
        raise ValueError("DequantizeLinear x_scale(shape=" + str(
            x_scale.shape) + ") and x_zero_point(shape=" + str(
                x_zero_point.shape) + ") must be in the same shape")
      if x_zero_point.dtype != x.dtype:
        raise ValueError(
            "DequantizeLinear x_zero_point(" + str(x_zero_point.dtype) +
            ") and x(" + str(x.dtype) + ") must be in the same dtype")

  @classmethod
  def version_10(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]
    x = tf.cast(x, tf.float32)
    x_scale = tensor_dict[node.inputs[1]]
    if len(node.inputs) == 3 and x.dtype != tf.int32:
      x_zero_point = tensor_dict[node.inputs[2]]
      x_zero_point = tf.cast(x_zero_point, tf.float32)
      x = tf.subtract(x, x_zero_point)

    y = tf.multiply(x, x_scale)

    return [y]

  @classmethod
  def version_13(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]
    x = tf.cast(x, tf.float32)
    x_scale = tensor_dict[node.inputs[1]]
    axis = node.attrs.get("axis", 1)

    x_shape = tf_shape(x)
    x_rank = len(x_shape)
    x_scale_shape = tf_shape(x_scale)
    x_scale_rank = len(x_scale_shape)

    # Reshape process is needed for per-axis dequantization
    # when scale is a 1-D tensor
    if x_scale_rank == 1:
      shape_broadcast = list([1 for _ in range(axis)] + [x_shape[axis]] +
                             [1 for _ in range(axis + 1, x_rank)])
      x_scale = tf.reshape(x_scale, shape_broadcast)

    if len(node.inputs) == 3 and x.dtype != tf.int32:
      x_zero_point = tensor_dict[node.inputs[2]]
      x_zero_point = tf.cast(x_zero_point, tf.float32)
      x_zero_point = tf.reshape(
          x_zero_point, shape_broadcast) if x_scale_rank == 1 else x_zero_point
      x = tf.subtract(x, x_zero_point)

    y = tf.multiply(x, x_scale)

    return [y]
