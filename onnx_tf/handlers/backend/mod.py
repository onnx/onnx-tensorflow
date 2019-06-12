import tensorflow as tf

from onnx_tf.common import exception
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from .math_mixin import ArithmeticMixin


@onnx_op("Mod")
class Mod(ArithmeticMixin, BackendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    unsupported_dtype = [
        tf.int8, tf.int16, tf.uint8, tf.uint16, tf.uint32, tf.uint64
    ]
    x = kwargs["tensor_dict"][node.inputs[0]]
    y = kwargs["tensor_dict"][node.inputs[1]]
    if x.dtype in unsupported_dtype:
      exception.OP_UNSUPPORTED_EXCEPT(
          "Mod Dividend in " + str(x.dtype), "Tensorflow")
    if y.dtype in unsupported_dtype:
      exception.OP_UNSUPPORTED_EXCEPT(
          "Mod Divisor in " + str(y.dtype), "Tensorflow")

  @classmethod
  def version_10(cls, node, **kwargs):
    fmod = node.attrs.get("fmod", 0)
    tf_func = tf.floormod
    if fmod == 1:
      tf_func = tf.truncatemod
    return [cls.make_tensor_from_onnx_node(node, tf_func=tf_func, **kwargs)]
