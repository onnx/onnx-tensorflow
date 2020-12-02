import tensorflow as tf

from onnx_tf.common import exception
from onnx_tf.common import sys_config
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
import onnx_tf.common.data_type as data_type


@onnx_op("Clip")
class Clip(BackendHandler):
  cast_map = {
      tf.uint8: tf.int32,
      tf.uint16: tf.int32,
      tf.uint32: tf.int64,
      tf.int8: tf.int32,
      tf.int16: tf.int32
  }
  supported_types = [
      tf.int32, tf.int64, tf.float16, tf.float32, tf.float64, tf.bfloat16
  ]

  @classmethod
  def args_check(cls, node, **kwargs):
    # update cast map based on the auto_cast config option
    cls.cast_map[tf.uint64] = tf.int64 if sys_config.auto_cast else None

    x = kwargs["tensor_dict"][node.inputs[0]]

    # throw an error if the data type is not natively supported by
    # Tensorflow, cannot be safely cast, and auto-cast option is False
    if x.dtype in cls.cast_map and cls.cast_map[x.dtype] is None:
      exception.DTYPE_NOT_CAST_EXCEPT(
          "Clip input " + node.inputs[0] + " with data type '" +
          data_type.tf_to_np_str(x.dtype) + "'",
          data_type.tf_to_np_str_list(cls.supported_types))

  @classmethod
  def _common(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]
    x_dtype = x.dtype

    if cls.SINCE_VERSION < 11:
      # min/max were required and passed as attributes
      clip_value_min = node.attrs.get("min", tf.reduce_min(x))
      clip_value_max = node.attrs.get("max", tf.reduce_max(x))
    else:
      # min/max are optional and passed as inputs
      clip_value_min = tensor_dict[node.inputs[1]] if len(
          node.inputs) > 1 and node.inputs[1] != "" else x_dtype.min
      clip_value_max = tensor_dict[node.inputs[2]] if len(
          node.inputs) > 2 and node.inputs[2] != "" else x_dtype.max

    # tf.clip_by_value doesn't support uint8, uint16, uint32, int8 and int16
    # dtype for x, therefore need to upcast it to tf.int32 or tf.int64
    need_cast = x_dtype in cls.cast_map
    x = tf.cast(x, cls.cast_map[x_dtype]) if need_cast else x
    clip_value_min = tf.cast(
        clip_value_min, cls.cast_map[x_dtype]) if need_cast else clip_value_min
    clip_value_max = tf.cast(
        clip_value_max, cls.cast_map[x_dtype]) if need_cast else clip_value_max
    y = tf.clip_by_value(x, clip_value_min, clip_value_max)
    y = tf.cast(y, x_dtype) if need_cast else y

    return [y]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_12(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
