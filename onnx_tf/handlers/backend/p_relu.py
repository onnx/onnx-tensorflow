import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from .broadcast_mixin import BroadcastMixin


@onnx_op("PRelu")
class PRelu(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    """
    Reference implementation at
    https://github.com/tflearn/tflearn/blob/4ba8c8d78bf1bbdfc595bf547bad30580cb4c20b/tflearn/activations.py#L191
    """
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]
    slope = BroadcastMixin.explicit_broadcast([x, tensor_dict[node.inputs[1]]])
    pos = tf.nn.relu(x)
    neg = slope * (x - abs(x)) * 0.5
    return [pos + neg]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_9(cls, node, **kwargs):
    return cls._common(node, **kwargs)
