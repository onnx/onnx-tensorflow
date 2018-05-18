from onnx_tf.common import exception
from onnx_tf.handlers.frontend_handler import FrontendHandler


class Split(FrontendHandler):
  _TF_OP = ["SplitV"]

  @classmethod
  def param_check(cls, node, version, **kwargs):
    if version == 2:
      if node.inputs[1] not in kwargs["consts"]:
        exception.CONST_NOT_FOUND_EXCEPT(node.inputs[1], node.op)
    if node.inputs[2] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[2], node.op)

  @classmethod
  def version_1(cls, node, **kwargs):
    consts = kwargs["consts"]
    axis = int(consts[node.inputs[2]])
    return cls.make_node(
        node, [node.inputs[0], node.inputs[1]],
        cls.get_outputs_names(node, num=node.attr["num_split"]),
        version=1,
        axis=axis)

  @classmethod
  def version_2(cls, node, **kwargs):
    consts = kwargs["consts"]
    split = consts[node.inputs[1]]
    axis = int(consts[node.inputs[2]])
    return cls.make_node(
        node, [node.inputs[0]],
        cls.get_outputs_names(node, num=node.attr["num_split"]),
        version=2,
        split=split,
        axis=axis)
