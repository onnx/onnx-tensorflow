import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from .math_mixin import BasicMathMixin


@onnx_op("Cos")
@tf_func(tf.cos)
class Cos(BasicMathMixin, BackendHandler):

  @classmethod
  def version_7(cls, node, **kwargs):
    return [cls.make_tf_tensor(node, **kwargs)]
