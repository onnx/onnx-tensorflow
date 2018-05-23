from onnx_tf.handlers.frontend_handler import FrontendHandler


class Relu(FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.make_node(node)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.make_node(node)
