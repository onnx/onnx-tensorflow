from onnx_tf.common import exception
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("Slice")
@tf_op("Slice")
class Slice(FrontendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    if node.inputs[1] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[1], node.op_type)
    if node.inputs[2] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[2], node.op_type)

  @classmethod
  def version_1(cls, node, **kwargs):
    consts = kwargs["consts"]
    return cls.make_node_from_tf_node(
        node, [node.inputs[0]],
        starts=consts[node.inputs[1]],
        ends=consts[node.inputs[1]] + consts[node.inputs[2]],
        axes=list(
            range(len(kwargs["node_dict"][node.inputs[0]].attr["shape"]))))
