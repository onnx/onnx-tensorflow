from onnx_tf.handlers.frontend_handler import FrontendHandler


class Reshape(FrontendHandler):

  @classmethod
  def param_check(cls, node, version, **kwargs):
    if version == 1:
      if node.inputs[1] not in kwargs["consts"]:
        raise RuntimeError("shape of Reshape is not found in graph consts.")

  @classmethod
  def version_1(cls, node, **kwargs):
    consts = kwargs["consts"]
    shape = consts[node.inputs[1]]
    return cls.make_node(node, [node.inputs[0]], [node.name], 1, shape=shape)

  @classmethod
  def version_5(cls, node, **kwargs):
    return cls.make_node(node, [node.inputs[0], node.inputs[1]], [node.name], 5)
