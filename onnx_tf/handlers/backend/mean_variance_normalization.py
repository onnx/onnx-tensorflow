import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("MeanVarianceNormalization")
class MeanVarianceNormalization(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    inputs = tensor_dict[node.inputs[0]]
    inputs_rank = inputs.shape.ndims

    across_channels = node.attrs.get("across_channels", 0)
    normalize_variance = node.attrs.get("normalize_variance", 1)

    moments_axes = [0] if not across_channels else [0, 1]
    moments_axes += list(range(inputs_rank))[2:]

    mean, variance = tf.nn.moments(inputs, moments_axes, keep_dims=True)

    if not normalize_variance:
      return [inputs - mean]
    return [(inputs - mean) / tf.sqrt(variance)]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)
