import copy
import tensorflow as tf

from onnx_tf.common import exception
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import partial_support
from onnx_tf.handlers.handler import ps_description
from onnx_tf.handlers.handler import tf_func


@onnx_op("OneHot")
@tf_func(tf.one_hot)
@partial_support(True)
@ps_description(
    "OneHot indices in uint16/uint32/uint64/int8/int16/" +
    "float16/float/double, or " +
    "OneHot depth in uint8/uint16/uint32/uint64/int8/" +
    "int16/int64/float16/float/double " + "are not supported in Tensorflow.")
class OneHot(BackendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    indices = tensor_dict[node.inputs[0]]
    depth = tensor_dict[node.inputs[1]]
    if indices.dtype not in [tf.uint8, tf.int32, tf.int64]:
      exception.OP_UNSUPPORTED_EXCEPT(
          "OneHot indices must be in uint8 or int32 or int64 " +
          "but it is currently in " + str(indices.dtype) + " which",
          "Tensorflow")
    if depth.dtype not in [tf.int32]:
      exception.OP_UNSUPPORTED_EXCEPT(
          "OneHot depth must be in int32 but it is currently in " +
          str(depth.dtype) + " which", "Tensorflow")

  @classmethod
  def _common(cls, node, **kwargs):
    attrs = copy.deepcopy(node.attrs)
    tensor_dict = kwargs["tensor_dict"]
    indices = tensor_dict[node.inputs[0]]
    depth = tensor_dict[node.inputs[1]]
    off_value = tensor_dict[node.inputs[2]][0]
    on_value = tensor_dict[node.inputs[2]][1]
    attrs["dtype"] = on_value.dtype
    return [
        cls.make_tensor_from_onnx_node(
            node,
            inputs=[indices, depth, on_value, off_value],
            attrs=attrs,
            **kwargs)
    ]

  @classmethod
  def version_9(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)
