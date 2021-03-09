import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.common import sys_config
from onnx_tf.common import exception
from onnx_tf.common import data_type
#import onnx_tf.common.data_type as data_type


@onnx_op("CumSum")
@tf_func(tf.math.cumsum)
class CumSum(BackendHandler):
  cast_map = {tf.uint32: tf.int64}
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
          "CumSum input " + node.inputs[0] + " with data type '" +
          data_type.tf_to_np_str(x.dtype) + "'",
          data_type.tf_to_np_str_list(cls.supported_types))

  @classmethod
  def _common(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]

    # handle data types that are not natively supported by Tensorflow
    dtype = x.dtype
    x = tf.cast(x, cls.cast_map[dtype]) if dtype in cls.cast_map else x
    inputs = [x]

    if len(node.inputs) > 1:
      # optional 0-D tensor, range [-rank(x), rank(x)-1]
      axis = kwargs["tensor_dict"][node.inputs[1]]
      inputs.append(axis)

    attrs = {
        "exclusive": bool(node.attrs.get("exclusive", 0)),
        "reverse": bool(node.attrs.get("reverse", 0))
    }

    result = cls.make_tensor_from_onnx_node(node, inputs=inputs, attrs=attrs)
    return [tf.cast(result, dtype) if dtype in cls.cast_map else result]

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_14(cls, node, **kwargs):
    return cls._common(node, **kwargs)
