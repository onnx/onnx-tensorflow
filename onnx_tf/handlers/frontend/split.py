from onnx_tf.common import exception
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.frontend_handler import version


class Split(FrontendHandler):
  TF_OP = ["SplitV"]

  @classmethod
  def param_check(cls, node, **kwargs):
    if cls.SINCE_VERSION == 2:
      if node.inputs[1] not in kwargs["consts"]:
        exception.CONST_NOT_FOUND_EXCEPT(node.inputs[1], node.op)
    if node.inputs[2] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[2], node.op)

  @classmethod
  @version(1)
  def version_1(cls, node, **kwargs):
    consts = kwargs["consts"]
    axis = int(consts[node.inputs[2]])
    return cls.make_node(
        node, [node.inputs[0], node.inputs[1]],
        cls.get_outputs_names(node, num=node.attr["num_split"]),
        axis=axis)

  @classmethod
  @version(2)
  def version_2(cls, node, **kwargs):
    consts = kwargs["consts"]
    split = consts[node.inputs[1]]
    axis = int(consts[node.inputs[2]])
    return cls.make_node(
        node, [node.inputs[0]],
        cls.get_outputs_names(node, num=node.attr["num_split"]),
        split=split,
        axis=axis)
