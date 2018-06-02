import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from .math_mixin import BasicMathMixin


@onnx_op("Pow")
@tf_func(tf.pow)
class Pow(BasicMathMixin, BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.limited_broadcast(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]
