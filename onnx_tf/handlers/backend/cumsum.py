import tensorflow as tf

from onnx_tf.common import exception
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import partial_support
from onnx_tf.handlers.handler import ps_description
from onnx_tf.handlers.handler import tf_func


@onnx_op("CumSum")
@tf_func(tf.math.cumsum)
@partial_support(True)
@ps_description(
    "CumSum inputs in uint32/uint64 " + "are not supported in Tensorflow."
)
class CumSum(BackendHandler):
  @classmethod
  def args_check(cls, node, **kwargs):
    supported_dtype = [
        tf.bfloat16, tf.half, tf.float32, tf.float64, tf.uint8, tf.uint16,
        tf.int8, tf.int16, tf.int32, tf.int64, tf.complex64, tf.complex128
    ]
    x = kwargs["tensor_dict"][node.inputs[0]]
    if x.dtype not in supported_dtype:
      exception.OP_UNSUPPORTED_EXCEPT(
          "CumSum input in " + str(x.dtype) + " which", "Tensorflow")

  @classmethod
  def version_11(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]
    inputs = [x]

    if len(node.inputs) > 1:
      # optional 0-D tensor, range [-rank(x), rank(x)-1]
      axis = tensor_dict[node.inputs[1]]
      inputs.append(axis)

    attrs = {
        "exclusive": bool(node.attrs.get("exclusive", 0)),
        "reverse": bool(node.attrs.get("reverse", 0))
    }

    return [cls.make_tensor_from_onnx_node(node, inputs=inputs, attrs=attrs)]
