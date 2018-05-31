import numpy as np

from onnx_tf.common import exception
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("ArgMin")
@tf_op("ArgMin")
class ArgMin(FrontendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    if node.inputs[1] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[1], node.op_type)

  @classmethod
  def version_1(cls, node, **kwargs):
    axis = np.asscalar(kwargs["consts"][node.inputs[1]])
    return cls.make_node_from_tf_node(
        node, [node.inputs[0]], axis=axis, keepdims=0)
