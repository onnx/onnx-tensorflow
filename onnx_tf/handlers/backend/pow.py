import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from .math_mixin import BasicMathMixin
from onnx_tf.common import sys_config
from onnx_tf.common import data_type


@onnx_op("Pow")
@tf_func(tf.pow)
class Pow(BasicMathMixin, BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):

    def x_cast(x):
      # Cast input to a data type that is supported by Tensorflow Pow
      cast_map = {tf.bfloat16: tf.float32}
      x = tf.cast(x, cast_map[x.dtype]) if x.dtype in cast_map else x
      return x

    def y_cast(y, to_dtype):
      # Cast exponent to the input data type when it is safe or auto cast is
      # set to True. Otherwise an error will occure due to the potential issue
      # in loss of data.
      if sys_config.auto_cast or data_type.is_safe_cast(y.dtype, to_dtype):
        return tf.cast(y, to_dtype)
      else:
        raise RuntimeError(
            "Exponent dtype cannot be safely cast to input dtype.")

    x = kwargs["tensor_dict"][node.inputs[0]]
    y = kwargs["tensor_dict"][node.inputs[1]]
    x_dtype = x.dtype
    y_dtype = y.dtype

    # Tensorflow Pow supports limited data types
    supported_types = [tf.float16, tf.float32, tf.float64, tf.int32, tf.int64]
    need_cast = x_dtype not in supported_types
    x = x_cast(x) if need_cast else x
    y = y_cast(y, x.dtype) if y_dtype != x.dtype else y

    inputs = [x, y]
    result = cls.make_tensor_from_onnx_node(node, inputs=inputs)

    return [tf.cast(result, x_dtype) if need_cast else result]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.limited_broadcast(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_12(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
