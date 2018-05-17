from onnx_tf.handlers.frontend_handler import FrontendHandler


class Concat(FrontendHandler):
  _TF_OP = ["ConcatV2"]

  @classmethod
  def param_check(cls, node, version, **kwargs):
    if node.inputs[-1] not in kwargs["consts"]:
      raise RuntimeError("axis of ConcatV2 is not found in graph consts.")

  @classmethod
  def version_1(cls, node, **kwargs):
    consts = kwargs["consts"]
    axis = int(consts[node.inputs[-1]])
    return cls.make_node(node, node.inputs[0:-1], version=1, axis=axis)

  @classmethod
  def version_4(cls, node, **kwargs):
    consts = kwargs["consts"]
    axis = int(consts[node.inputs[-1]])
    return cls.make_node(node, node.inputs[0:-1], version=4, axis=axis)
