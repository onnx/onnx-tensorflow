import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from .math_mixin import BasicMathMixin
from onnx_tf.common import sys_config
from onnx_tf.common import exception
from onnx_tf.common import data_type


@onnx_op("Pow")
@tf_func(tf.pow)
class Pow(BasicMathMixin, BackendHandler):
  x_cast_map = {tf.bfloat16: tf.float32}
  y_cast_map = {
      tf.uint8: tf.int16,
      tf.uint16: tf.int32,
      tf.uint32: tf.int64,
      tf.int8: tf.int16,
      tf.int16: tf.int32
  }
  supported_types = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]

  @classmethod
  def args_check(cls, node, **kwargs):
    # update cast map based on the auto_cast config option
    cls.y_cast_map[tf.uint64] = tf.int64 if sys_config.auto_cast else None

    y = kwargs["tensor_dict"][node.inputs[1]]

    # throw an error if the data type is not natively supported by
    # Tensorflow, cannot be safely cast, and auto-cast option is False
    if y.dtype in cls.y_cast_map and cls.y_cast_map[y.dtype] is None:
      exception.DTYPE_NOT_CAST_EXCEPT(
          "Pow input " + node.inputs[1] + " with data type '" +
          data_type.tf_to_np_str(y.dtype) + "'",
          data_type.tf_to_np_str_list(cls.supported_types))

  @classmethod
  def _common(cls, node, **kwargs):

    def x_cast(x):
      # Cast input to a data type that is supported by Tensorflow Pow
      return tf.cast(x, cls.x_cast_map[x.dtype])

    def y_cast(y, to_dtype):
      # Cast exponent to the input data type
      return tf.cast(y, to_dtype)

    x = kwargs["tensor_dict"][node.inputs[0]]
    y = kwargs["tensor_dict"][node.inputs[1]]
    x_dtype = x.dtype
    y_dtype = y.dtype

    # Tensorflow Pow supports limited data types
    x = x_cast(x) if x_dtype in cls.x_cast_map else x
    y = y_cast(y, x.dtype) if y_dtype != x.dtype else y

    inputs = [x, y]
    result = cls.make_tensor_from_onnx_node(node, inputs=inputs)

    return [tf.cast(result, x_dtype) if x_dtype in cls.x_cast_map else result]

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
