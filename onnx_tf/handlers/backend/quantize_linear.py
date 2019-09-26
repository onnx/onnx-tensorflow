import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("QuantizeLinear")
class QuantizeLinear(BackendHandler):

  @classmethod
  def version_10(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]
    y_scale = tensor_dict[node.inputs[1]]
    y_zero_point = tensor_dict[
        node.inputs[2]] if node.inputs[2] in tensor_dict else tf.zeros(
            (), tf.uint8)
    y_dtype = y_zero_point.dtype

    x = tf.cast(x, tf.float32)
    y = tf.divide(x, y_scale)
    y = tf.round(y)
    y_zero_point = tf.cast(y_zero_point, tf.float32)
    y = tf.add(y, y_zero_point)
    y = tf.saturate_cast(y, y_dtype)

    return [y]
