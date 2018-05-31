import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.handlers.handler import tf_op
from .math_mixin import BasicMathMixin


@onnx_op("Ceil")
@tf_op("Ceil")
@tf_func(tf.ceil)
class Ceil(BasicMathMixin, BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return [cls.make_tf_tensor(node, **kwargs)]

  @classmethod
  def version_6(cls, node, **kwargs):
    return [cls.make_tf_tensor(node, **kwargs)]
