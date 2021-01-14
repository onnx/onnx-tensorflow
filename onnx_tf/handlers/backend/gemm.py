import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.common import sys_config
from onnx_tf.common import exception
import onnx_tf.common.data_type as data_type


@onnx_op("Gemm")
class Gemm(BackendHandler):
  cast_map = {}
  supported_types = [
      tf.float32
  ]

  @classmethod
  def args_check(cls, node, **kwargs):
    # update cast map based on the auto_cast config option
    cls.cast_map[tf.float16] = tf.float32 if sys_config.auto_cast else None
    cls.cast_map[tf.float64] = tf.float32 if sys_config.auto_cast else None
    cls.cast_map[tf.uint32] = tf.float32 if sys_config.auto_cast else None
    cls.cast_map[tf.uint64] = tf.float32 if sys_config.auto_cast else None
    cls.cast_map[tf.int32] = tf.float32 if sys_config.auto_cast else None
    cls.cast_map[tf.int64] = tf.float32 if sys_config.auto_cast else None
    cls.cast_map[tf.bfloat16] = tf.float32 if sys_config.auto_cast else None

    x = kwargs["tensor_dict"][node.inputs[0]]

    # throw an error if the data type is not natively supported by
    # Tensorflow, cannot be safely cast, and auto_cast option is False
    if x.dtype in cls.cast_map and cls.cast_map[x.dtype] is None:
      exception.DTYPE_NOT_CAST_EXCEPT(
          "Gemm input " + node.inputs[0] + " with data type '" +
          data_type.tf_to_np_str(x.dtype) + "'",
          data_type.tf_to_np_str_list(cls.supported_types))

  @classmethod
  def _common(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]
    dtype = x.dtype
    x = tf.keras.layers.Flatten()(x)
    # The Flatten API changes data type from tf.float64 to tf.float32
    # so we need the following line to get the original type back
    x = tf.cast(x, dtype) if dtype is tf.float64 else x
    y = tensor_dict[node.inputs[1]]

    if len(node.inputs) > 2:
      z = tensor_dict[node.inputs[2]]
    else:
      z = 0

    if node.attrs.get("transA", 0):
      x = tf.transpose(x)
    if node.attrs.get("transB", 0):
      y = tf.transpose(y)
    alpha = node.attrs.get("alpha", 1.0)
    beta = node.attrs.get("beta", 1.0)

    # We cast to either input or attribute data type to preserve precision
    if dtype in [tf.float64]:
      # cast to input data type
      alpha = tf.cast(alpha, dtype)
      beta = tf.cast(beta, dtype)
      return [alpha * tf.matmul(x, y) + beta * z]
    else:
      # cast to attribute data type
      x = tf.cast(x, cls.cast_map[dtype]) if dtype in cls.cast_map else x
      y = tf.cast(y, cls.cast_map[dtype]) if dtype in cls.cast_map else y
      z = tf.cast(z, cls.cast_map[dtype]) if dtype in cls.cast_map else z
      result = alpha * tf.matmul(x, y) + beta * z
      return [tf.cast(result, dtype) if dtype in cls.cast_map else result]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_9(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
