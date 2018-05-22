from onnx_tf.handlers.frontend_handler import FrontendHandler
from .control_flow_mixin import ComparisonMixin


class Equal(ComparisonMixin, FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.comparison_op(node, **kwargs)
