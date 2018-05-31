import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.handlers.handler import tf_op
from .control_flow_mixin import LogicalMixin


@onnx_op("And")
@tf_op("LogicalAnd")
@tf_func(tf.logical_and)
class Add(LogicalMixin, BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return [cls.make_tf_tensor(node, **kwargs)]

  @classmethod
  def version_7(cls, node, **kwargs):
    return [cls.make_tf_tensor(node, **kwargs)]
