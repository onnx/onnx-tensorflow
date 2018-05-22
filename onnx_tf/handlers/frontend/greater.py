from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.frontend_handler import version
from .control_flow_mixin import ComparisonMixin


class Greater(ComparisonMixin, FrontendHandler):

  @classmethod
  @version(1)
  def version_1(cls, node, **kwargs):
    return cls.comparison_op(node, **kwargs)
