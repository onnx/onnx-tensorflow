import numpy as np

from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("Gather")
@tf_op("GatherV2")
class Gather(FrontendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    if len(node.inputs) == 3 and node.inputs[2] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[2], node.op_type)

  @classmethod
  def version_1(cls, node, **kwargs):
    consts = kwargs["consts"]
    axis = np.asscalar(consts[node.inputs[2]]) if len(node.inputs) == 3 else 0
    return cls.make_node_from_tf_node(node, node.inputs[0:2], axis=axis)
