from onnx_tf.common import exception
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("TopK")
@tf_op("TopKV2")
class TopK(FrontendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    if node.inputs[1] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[1], node.op_type)

  @classmethod
  def version_1(cls, node, **kwargs):
    consts = kwargs["consts"]
    k = int(consts[node.inputs[1]])
    return cls.make_node_from_tf_node(
        node, inputs=[node.inputs[0]], k=k, axis=-1)
