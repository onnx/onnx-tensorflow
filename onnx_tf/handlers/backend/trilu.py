import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.common.tf_helper import tf_shape


@onnx_op("Trilu")
class Trilu(BackendHandler):

  @classmethod
  def version_14(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    if len(node.inputs) >= 2:
      k = kwargs["tensor_dict"][node.inputs[1]]
      #handle pos out
      x_shape = tf_shape(x, dtype=k.dtype)
      if k > x_shape[-1]:
        k = x_shape[-1]
      elif k < 0 - x_shape[-2]:
        k = 0 - x_shape[-2]
    else:
      k = tf.constant(0, dtype=tf.int64)
    keep_triangle = tf.constant(-1, dtype=k.dtype)
    upper = node.attrs.get("upper", 1)

    if upper == 1:
      if k > 0:
        return [tf.subtract(x, tf.linalg.band_part(x, keep_triangle, k - 1))]
      else:
        return [tf.linalg.band_part(x, -k, keep_triangle)]
    else:
      if k >= 0:
        return [tf.linalg.band_part(x, keep_triangle, k)]
      else:
        return [tf.subtract(x, tf.linalg.band_part(x, -1 - k, keep_triangle))]
