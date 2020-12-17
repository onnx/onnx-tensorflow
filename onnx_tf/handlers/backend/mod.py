import tensorflow as tf

from onnx_tf.common import exception
from onnx_tf.common import sys_config
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from .math_mixin import ArithmeticMixin
import onnx_tf.common.data_type as data_type


@onnx_op("Mod")
class Mod(ArithmeticMixin, BackendHandler):
  cast_map = {
      tf.uint8: tf.int32,
      tf.uint16: tf.int32,
      tf.uint32: tf.int64,
      tf.int8: tf.int32,
      tf.int16: tf.int32,
      tf.float16: tf.float32
  }
  supported_types = [
      tf.int32, tf.int64, tf.float32, tf.float64, tf.bfloat16
  ]

  @classmethod
  def args_check(cls, node, **kwargs):
    # update cast map based on the auto_cast config option
    cls.cast_map[tf.uint64] = tf.int64 if sys_config.auto_cast else None

    x = kwargs["tensor_dict"][node.inputs[0]]
    y = kwargs["tensor_dict"][node.inputs[1]]

    # throw an error if the data type is not natively supported by
    # Tensorflow, cannot be safely cast, and auto-cast option is False
    if x.dtype in cls.cast_map and cls.cast_map[x.dtype] is None:
      exception.DTYPE_NOT_CAST_EXCEPT(
          "Mod input " + node.inputs[0] + " with data type '" +
          data_type.tf_to_np_str(x.dtype) + "'",
          data_type.tf_to_np_str_list(cls.supported_types))

    # throw an error if inputs A and B are not in the same data type
    if x.dtype != y.dtype:
      exception.OP_UNSUPPORTED_EXCEPT("Mod with inputs in different data types",
                                      "Tensorflow")

  @classmethod
  def _common(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    y = kwargs["tensor_dict"][node.inputs[1]]
    x_dtype = x.dtype
    y_dtype = y.dtype
    fmod = node.attrs.get("fmod", 0)

    # cast inputs if not natively support by Tensorflow API
    need_cast = x_dtype in cls.cast_map
    x = tf.cast(x, cls.cast_map[x_dtype]) if need_cast else x
    y = tf.cast(y, cls.cast_map[y_dtype]) if need_cast else y

    tf_func = tf.truncatemod if fmod == 1 else tf.math.floormod
    z = cls.make_tensor_from_onnx_node(node,
                                       tf_func=tf_func,
                                       inputs=[x, y],
                                       **kwargs)
    z = tf.cast(z, x_dtype) if need_cast else z

    return [z]

  @classmethod
  def version_10(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
