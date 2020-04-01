import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Clip")
@tf_func(tf.clip_by_value)
class Clip(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]

    if cls.SINCE_VERSION < 11:
      # min/max were required and passed as attributes
      clip_value_min = node.attrs.get("min", tf.reduce_min(x))
      clip_value_max = node.attrs.get("max", tf.reduce_max(x))
    else:
      # min/max are optional and passed as inputs
      clip_value_min = tensor_dict[
          "min"] if "min" in tensor_dict else x.dtype.min
      clip_value_max = tensor_dict[
          "max"] if "max" in tensor_dict else x.dtype.max

    return [
        cls.make_tensor_from_onnx_node(
            node, inputs=[x, clip_value_min, clip_value_max])
    ]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)
