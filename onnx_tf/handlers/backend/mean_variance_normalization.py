import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("MeanVarianceNormalization")
class MeanVarianceNormalization(BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    inputs = tensor_dict[node.inputs[0]]
    inputs_rank = inputs.shape.ndims

    across_channels = node.attrs.get("across_channels", 0)
    normalize_variance = node.attrs.get("normalize_variance", 1)

    moments_axes = [0] if not across_channels else [0, 1]
    moments_axes += list(range(inputs_rank))[2:]

    mean, variance = tf.nn.moments(inputs, moments_axes, keepdims=True)

    if not normalize_variance:
      return [inputs - mean]
    return [(inputs - mean) / tf.sqrt(variance)]

  @classmethod
  def _common(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    inputs = tensor_dict[node.inputs[0]]
    inputs_rank = inputs.shape.ndims
    # To satisfy default axes=[0,2,3], also assume the
    # following default when rank is not 4
    # rank1 -> axes=[0]
    # rank2 -> axes=[0]
    # rank3 -> axes=[0,2]
    # rank4 -> axes=[0,2,3]
    # rankN -> axes=[0,2,3,..,N-1]
    # TODO(tedhtchang): Since input tensor is no longer limited
    # to shape [N,C,H,W], consider using "[0]" or "[]" as default axes.
    # See issue https://github.com/onnx/onnx/issues/2047
    default_axes = [0] if inputs_rank < 3 else [0, 2]
    default_axes += list(range(inputs_rank))[3:]
    moments_axes = node.attrs.get("axes", default_axes)
    mean, variance = tf.nn.moments(inputs, moments_axes, keepdims=True)
    return [(inputs - mean) / tf.sqrt(variance)]

  @classmethod
  def version_9(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)

