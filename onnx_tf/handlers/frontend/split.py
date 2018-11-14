from onnx_tf.common import exception
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("Split")
@tf_op(["SplitV", "Split"])
class Split(FrontendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    if node.op_type == "SplitV":
      if node.inputs[2] not in kwargs["consts"]:
        exception.CONST_NOT_FOUND_EXCEPT(node.inputs[2], node.op_type)
    if node.op_type == "Split":
      if node.inputs[0] not in kwargs["consts"]:
        exception.CONST_NOT_FOUND_EXCEPT(node.inputs[0], node.op_type)

  @classmethod
  def version_1(cls, node, **kwargs):
    consts = kwargs["consts"]
    if node.op_type == "SplitV":
      axis = int(consts[node.inputs[2]])
      x = [node.inputs[0], node.attr["num_split"]]
    elif node.op_type == "Split":
      axis = int(consts[node.inputs[0]])
      x = [node.inputs[1], node.attr["num_split"]]
    return cls.make_node_from_tf_node(
        node, x, node.get_outputs_names(num=node.attr["num_split"]), axis=axis)

  @classmethod
  def version_2(cls, node, **kwargs):
    consts = kwargs["consts"]
    if node.op_type == "SplitV":
      axis = int(consts[node.inputs[2]])
      x = [node.inputs[0]]
    elif node.op_type == "Split":
      axis = int(consts[node.inputs[0]])
      x = [node.inputs[1]]
    return cls.make_node_from_tf_node(
        node,
        x,
        node.get_outputs_names(num=node.attr["num_split"]),
        split=[shape[axis] for shape in node.attr["_output_shapes"]],
        axis=axis)
