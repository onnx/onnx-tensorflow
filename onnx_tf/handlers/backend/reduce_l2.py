import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from .math_mixin import ReductionMixin


@onnx_op("ReduceL2")
@tf_func(tf.norm)
class ReduceL2(ReductionMixin, BackendHandler):

  @classmethod
  def process_attrs(cls, attrs):
    return cls._process_attrs(attrs, default={"ord": 2})

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)
