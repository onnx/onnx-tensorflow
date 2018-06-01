import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from .control_flow_mixin import LogicalMixin
from .broadcast_mixin import BroadcastMixin


@onnx_op("Xor")
@tf_func(tf.logical_xor)
class Xor(LogicalMixin, BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return [BroadcastMixin.limited_broadcast(node, **kwargs)]

  @classmethod
  def version_7(cls, node, **kwargs):
    return [cls.make_tf_tensor(node, **kwargs)]
