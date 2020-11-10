import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op

@onnx_op("Einsum")

class Einsum(BackendHandler):

  @classmethod
  def version_12(cls, node, **kwargs):
    equation = node.attrs.get("equation", "")
    inputs = [kwargs["tensor_dict"][inp] for inp in node.inputs]
    return [tf.einsum(equation, *inputs)]
