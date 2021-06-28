import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("Trilu")
class Trilu(BackendHandler):

  @classmethod
  def version_14(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    k = tf.constant(0, dtype=tf.int32)
    if len(node.inputs) >= 2:
        k = kwargs["tensor_dict"][node.inputs[1]]
    keep_triangle = tf.constant(-1, dtype=k.dtype)
    upper = node.attrs.get("upper", 1)
    
    #handle pos out
    if k > x.shape[-1]:
        k = tf.constant(x.shape[-1], dtype=k.dtype)
    elif k < 0-x.shape[-2]:
        k = tf.constant(0-x.shape[-2], dtype=k.dtype)
    
    if upper == 1:
        if k > 0:
            return [tf.subtract(x,tf.linalg.band_part(x,keep_triangle,k-1))]
        else:
            return [tf.linalg.band_part(x, -k, keep_triangle)]
    else:
        if k >= 0:
            return [tf.linalg.band_part(x,keep_triangle,k)]
        else:
            return [tf.subtract(x,tf.linalg.band_part(x, -1-k, keep_triangle))]

