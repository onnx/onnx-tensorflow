import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.handlers.handler import tf_op
from .math_mixin import BasicMathMixin


@onnx_op("Acos")
@tf_op("Acos")
@tf_func(tf.acos)
class Acos(BasicMathMixin, BackendHandler):

  @classmethod
  def version_7(cls, node, **kwargs):
    return [cls.make_tf_tensor(node, **kwargs)]
