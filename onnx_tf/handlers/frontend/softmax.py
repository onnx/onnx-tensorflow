from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.frontend_handler import version


class Softmax(FrontendHandler):

  @classmethod
  @version(1)
  def version_1(cls, node, **kwargs):
    return cls.make_node(node, axis=1)
