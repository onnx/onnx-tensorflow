import tensorflow as tf

from onnx_tf.common import exception
from onnx_tf.common import data_type
from onnx_tf.common import sys_config
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("MatMul")
@tf_func(tf.matmul)
class MatMul(BackendHandler):
  supported_types = [
      tf.bfloat16, tf.float16, tf.float32, tf.float64, tf.int32, tf.int64
  ]
  cast_map = {tf.uint32: tf.int64}

  @classmethod
  def args_check(cls, node, **kwargs):
    # update cast map based on the auto_cast config option
    cls.cast_map[tf.uint64] = tf.int64 if sys_config.auto_cast else None

    dtype = kwargs["tensor_dict"][node.inputs[0]].dtype
    if dtype in cls.cast_map and cls.cast_map[dtype] is None:
      exception.DTYPE_NOT_CAST_EXCEPT(
          "MatMul input " + node.inputs[0] + " with data type '" +
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
    return cls._common(node, **kwargs)

  @classmethod
  def version_9(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
