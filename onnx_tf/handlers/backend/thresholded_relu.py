import warnings

import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("ThresholdedRelu")
class ThresholdedRelu(BackendHandler):

  @classmethod
  def process_attrs(cls, attrs):
    return cls._process_attrs(attrs, rename={"alpha": "theta"})

  @classmethod
  def version_1(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    if not "alpha" in node.attrs.keys():
      warnings.warn("Provide an alpha value.", UserWarning)
      alpha = 1
    else:
      alpha = node.attrs["alpha"]

    epsilon = 1e-5
    return [tf.nn.relu(x) - tf.nn.relu(tf.sign(alpha - x + epsilon) * x)]
