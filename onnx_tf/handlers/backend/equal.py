import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from .broadcast_mixin import BroadcastMixin
from .control_flow_mixin import ComparisonMixin


@onnx_op("Equal")
@tf_func(tf.equal)
class Equal(BroadcastMixin, ComparisonMixin, BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.limited_broadcast(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return [cls.make_tf_tensor(node, **kwargs)]
