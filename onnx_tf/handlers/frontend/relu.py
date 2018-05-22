from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.frontend_handler import version


class Relu(FrontendHandler):

  @classmethod
  @version(1)
  def version_1(cls, node, **kwargs):
    return cls.make_node(node)

  @classmethod
  @version(6)
  def version_6(cls, node, **kwargs):
    return cls.make_node(node)
