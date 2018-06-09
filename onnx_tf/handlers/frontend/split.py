from onnx_tf.common import exception
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("Split")
@tf_op("SplitV")
class Split(FrontendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    if cls.SINCE_VERSION == 2:
      if node.inputs[1] not in kwargs["consts"]:
        exception.CONST_NOT_FOUND_EXCEPT(node.inputs[1], node.op_type)
    if node.inputs[2] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[2], node.op_type)

  @classmethod
  def version_1(cls, node, **kwargs):
    consts = kwargs["consts"]
    axis = int(consts[node.inputs[2]])
    return cls.make_node_from_tf_node(
        node, [node.inputs[0], node.inputs[1]],
        cls.get_outputs_names(node, num=node.attr["num_split"]),
        axis=axis)

  @classmethod
  def version_2(cls, node, **kwargs):
    consts = kwargs["consts"]
    split = consts[node.inputs[1]]
    axis = int(consts[node.inputs[2]])
    return cls.make_node_from_tf_node(
        node, [node.inputs[0]],
        cls.get_outputs_names(node, num=node.attr["num_split"]),
        split=split,
        axis=axis)
