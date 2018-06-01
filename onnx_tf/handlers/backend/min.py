import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from .math_mixin import ReductionMixin


@onnx_op("Min")
@tf_func(tf.reduce_min)
class Min(ReductionMixin, BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return [cls._common(node, **kwargs)]

  @classmethod
  def version_6(cls, node, **kwargs):
    return [cls._common(node, **kwargs)]
