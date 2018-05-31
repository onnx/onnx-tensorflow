import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.handlers.handler import tf_op
from .math_mixin import BasicMathMixin


@onnx_op("Abs")
@tf_op("Abs")
@tf_func(tf.abs)
class Abs(BasicMathMixin, BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return [cls.make_tf_tensor(node, **kwargs)]

  @classmethod
  def version_6(cls, node, **kwargs):
    return [cls.make_tf_tensor(node, **kwargs)]