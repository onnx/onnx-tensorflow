import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from .control_flow_mixin import ComparisonMixin
from onnx_tf.common import sys_config
from onnx_tf.common import exception
import onnx_tf.common.data_type as data_type


@onnx_op("LessOrEqual")
@tf_func(tf.less_equal)
class LessOrEqual(ComparisonMixin, BackendHandler):
  cast_map = {tf.uint16: tf.int32, tf.uint32: tf.int64}
  supported_types = [
      tf.uint8, tf.int8, tf.int16, tf.int32, tf.int64, tf.float16,
      tf.float32, tf.float64, tf.bfloat16
  ]

  @classmethod
  def args_check(cls, node, **kwargs):
    # update cast map based on the auto_cast config option
    cls.cast_map[tf.uint64] = tf.int64 if sys_config.auto_cast else None

    x = kwargs["tensor_dict"][node.inputs[0]]
    y = kwargs["tensor_dict"][node.inputs[1]]

    # throw an error if the data type is not natively supported by
    # Tensorflow, cannot be safely cast, and auto_cast option is False
    if x.dtype in cls.cast_map and cls.cast_map[x.dtype] is None:
      exception.DTYPE_NOT_CAST_EXCEPT(
          "LessOrEqual input " + node.inputs[0] + " with data type '" +
          data_type.tf_to_np_str(x.dtype) + "'",
          data_type.tf_to_np_str_list(cls.supported_types))
    if y.dtype in cls.cast_map and cls.cast_map[y.dtype] is None:
      exception.DTYPE_NOT_CAST_EXCEPT(
          "LessOrEqual input " + node.inputs[1] + " with data type '" +
          data_type.tf_to_np_str(y.dtype) + "'",
          data_type.tf_to_np_str_list(cls.supported_types))

  @classmethod
  def version_12(cls, node, **kwargs):
    def dtype_cast(x):
      return tf.cast(x, cls.cast_map[x.dtype]) if x.dtype in cls.cast_map else x

    # handle data types that are not natively supported by Tensorflow
    x = dtype_cast(kwargs["tensor_dict"][node.inputs[0]])
    y = dtype_cast(kwargs["tensor_dict"][node.inputs[1]])

    return [cls.make_tensor_from_onnx_node(node, inputs=[x, y])]
