import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.handlers.handler import tf_op
from .math_mixin import ArithmeticMixin
from .broadcast_mixin import BroadcastMixin


@onnx_op("Div")
@tf_op("RealDiv")
@tf_func(tf.div)
class Div(ArithmeticMixin, BackendHandler):

  @classmethod
  def _limited_broadcast(cls, node, **kwargs):
    if node.attrs.get("broadcast") == 1:
      inputs = BroadcastMixin.explicit_broadcast(node.inputs,
                                                 node.attrs.get("axis", None),
                                                 kwargs["tensor_dict"])
      return [cls.make_tf_tensor(node, inputs=inputs, **kwargs)]
    return [cls.make_tf_tensor(node, **kwargs)]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._limited_broadcast(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls._limited_broadcast(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return [cls.make_tf_tensor(node, **kwargs)]
