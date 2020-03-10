import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("BitShift")
class BitShift(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    tf_func = tf.bitwise.left_shift if node.attrs.get(
        "direction") == "LEFT" else tf.bitwise.right_shift
    return [
        cls.make_tensor_from_onnx_node(node, tf_func=tf_func, **kwargs)
    ]

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)
