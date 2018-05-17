from onnx_tf.handlers.frontend_handler import FrontendHandler


class LogicalCommon(FrontendHandler):

  @classmethod
  def logical_op(cls, node, version, **kwargs):
    return cls.make_node(node, node.inputs, [node.name], version, broadcast=1)
