from onnx_tf.handlers.frontend_handler import FrontendHandler


class Softplus(FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.make_node(node)
