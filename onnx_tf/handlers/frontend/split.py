from onnx_tf.handlers.frontend_handler import FrontendHandler


class Split(FrontendHandler):
  _TF_OP = ["SplitV"]

  @classmethod
  def param_check(cls, node, version, **kwargs):
    if version == 2:
      if node.inputs[1] not in kwargs["consts"]:
        raise RuntimeError(
            "num_or_size_splits of SplitV is not found in graph consts.")
    if node.inputs[2] not in kwargs["consts"]:
      raise RuntimeError("axis of SplitV is not found in graph consts.")

  @classmethod
  def version_1(cls, node, **kwargs):
    consts = kwargs["consts"]
    axis = int(consts[node.inputs[2]])
    output_names = [
        node.name + ":{}".format(i) if i > 0 else node.name
        for i in range(node.attr["num_split"])
    ]
    return cls.make_node(
        node, [node.inputs[0], node.inputs[1]],
        output_names,
        version=1,
        axis=axis)

  @classmethod
  def version_2(cls, node, **kwargs):
    consts = kwargs["consts"]
    split = consts[node.inputs[1]]
    axis = int(consts[node.inputs[2]])
    output_names = [
        node.name + ":{}".format(i) if i > 0 else node.name
        for i in range(node.attr["num_split"])
    ]
    return cls.make_node(
        node, [node.inputs[0]], output_names, version=2, split=split, axis=axis)
