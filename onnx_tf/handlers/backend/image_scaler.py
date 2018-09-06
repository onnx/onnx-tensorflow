import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("ImageScaler")
class ImageScaler(BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    input_dict = kwargs["tensor_dict"]
    x = input_dict[node.inputs[0]]
    scale = node.attrs.get("scale", 1.0)
    output = tf.multiply(x, scale)
    if "bias" in node.attrs:
      bias = node.attrs["bias"]
      output = tf.nn.bias_add(output, bias, data_format="NCHW")
    return [output]
