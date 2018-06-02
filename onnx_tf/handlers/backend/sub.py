import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from .broadcast_mixin import BroadcastMixin
from .math_mixin import ArithmeticMixin


@onnx_op("Sub")
@tf_func(tf.subtract)
class Sub(ArithmeticMixin, BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._limited_broadcast(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls._limited_broadcast(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return [cls.make_tf_tensor(node, **kwargs)]
