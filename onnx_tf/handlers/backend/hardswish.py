import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("HardSwish")
class HardSwish(BackendHandler):

  @classmethod
  def version_14(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    dtype = x.dtype
    alpha = 1 / 6
    beta = 0.5
    return [
        x *
        tf.maximum(tf.constant(0, dtype=dtype),
                   tf.minimum(tf.constant(1, dtype=dtype), alpha * x + beta))
    ]
