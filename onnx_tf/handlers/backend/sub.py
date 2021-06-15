import tensorflow as tf

from onnx_tf.common import exception
from onnx_tf.common import data_type
from onnx_tf.common import sys_config
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from .math_mixin import ArithmeticMixin


@onnx_op("Sub")
@tf_func(tf.subtract)
class Sub(ArithmeticMixin, BackendHandler):
  supported_types = [
      tf.bfloat16, tf.float16, tf.float32, tf.float64, tf.int32, tf.int64,
      tf.uint32, tf.int8, tf.int16, tf.uint8, tf.uint16
  ]
  cast_map = {}
  cast_map[tf.uint64] = tf.int64 if sys_config.auto_cast else None

  @classmethod
  def args_check(cls, node, **kwargs):
    dtype = kwargs["tensor_dict"][node.inputs[0]].dtype
    if dtype in cls.cast_map and cls.cast_map[dtype] is None:
      exception.DTYPE_NOT_CAST_EXCEPT(
          "Sub input " + node.inputs[0] + " with data type '" +
          data_type.tf_to_np_str(dtype) + "'",
          data_type.tf_to_np_str_list(cls.supported_types))

  @classmethod
  def _common(cls, node, **kwargs):
    values = [kwargs["tensor_dict"][inp] for inp in node.inputs]
    dtype = values[0].dtype
    if dtype in cls.cast_map:
      values = [tf.cast(v, cls.cast_map[v.dtype]) for v in values]
    result = cls.make_tensor_from_onnx_node(node, inputs=values, **kwargs)
    return [tf.cast(result, dtype) if dtype in cls.cast_map else result]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.limited_broadcast(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.limited_broadcast(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_14(cls, node, **kwargs):
    return cls._common(node, **kwargs)
