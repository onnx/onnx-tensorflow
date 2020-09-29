import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from .control_flow_mixin import ComparisonMixin


@onnx_op("GreaterOrEqual")
@tf_func(tf.greater_equal)
class GreaterOrEqual(ComparisonMixin, BackendHandler):

  @classmethod
  def version_12(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]
