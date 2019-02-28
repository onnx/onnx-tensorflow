from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op
from .control_flow_mixin import ComparisonMixin


@onnx_op("Greater")
@tf_op("Greater")
class Greater(ComparisonMixin, FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.comparison_op(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls.comparison_op(node, **kwargs)

  @classmethod
  def version_9(cls, node, **kwargs):
    return cls.comparison_op(node, **kwargs)
