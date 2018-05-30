import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.handlers.handler import tf_op
from .math_mixin import ArithmeticMixin


@onnx_op("Add")
@tf_op("Add")
@tf_func(tf.add)
class Add(ArithmeticMixin, BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return [cls.make_tf_node(node, **kwargs)]

  @classmethod
  def version_6(cls, node, **kwargs):
    return [cls.make_tf_node(node, **kwargs)]

  @classmethod
  def version_7(cls, node, **kwargs):
    return [cls.make_tf_node(node, **kwargs)]
