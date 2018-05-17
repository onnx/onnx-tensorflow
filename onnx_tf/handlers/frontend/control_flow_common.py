from onnx_tf.handlers.frontend_handler import FrontendHandler


class LogicalCommon(FrontendHandler):

  @classmethod
  def logical_op(cls, node, version, broadcast=1, **kwargs):
    ex_kwargs = {}
    if broadcast == 1:
      ex_kwargs["broadcast"] = 1
    return cls.make_node(node, version=version, **ex_kwargs)
