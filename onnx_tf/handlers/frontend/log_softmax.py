from onnx_tf.handlers.frontend_handler import FrontendHandler


class LogSoftmax(FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.make_node(node, version=1)
