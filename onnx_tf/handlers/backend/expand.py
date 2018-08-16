import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("Expand")
class Expand(BackendHandler):

  @classmethod
  def version_8(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x, shape = tensor_dict[node.inputs[0]], tensor_dict[node.inputs[1]]
    ones = tf.ones(shape, dtype=x.dtype)
    return [x * ones]
