import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from .math_mixin import BasicMathMixin
from onnx_tf.common import sys_config
from onnx_tf.common import exception
from onnx_tf.common import data_type


@onnx_op("Sign")
@tf_func(tf.sign)
class Sign(BasicMathMixin, BackendHandler):
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
          "Sign input " + node.inputs[0] + " with data type '" +
          data_type.tf_to_np_str(x.dtype) + "'",
          data_type.tf_to_np_str_list(cls.supported_types))

  @classmethod
  def _common(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]

    # handle data types that are not natively supported by Tensorflow
    dtype = x.dtype
    inputs = [tf.cast(x, cls.cast_map[dtype]) if dtype in cls.cast_map else x]

    result = cls.make_tensor_from_onnx_node(node, inputs=inputs)
    return [tf.cast(result, dtype) if dtype in cls.cast_map else result]

  @classmethod
  def version_9(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
