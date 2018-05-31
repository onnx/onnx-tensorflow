from onnx_tf.common import exception
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("Reshape")
@tf_op("Reshape")
class Reshape(FrontendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    if cls.SINCE_VERSION == 1:
      if node.inputs[1] not in kwargs["consts"]:
        exception.CONST_NOT_FOUND_EXCEPT(node.inputs[1], node.op_type)

  @classmethod
  def version_1(cls, node, **kwargs):
    consts = kwargs["consts"]
    shape = consts[node.inputs[1]]
    return cls.make_node_from_tf_node(node, [node.inputs[0]], shape=shape)

  @classmethod
  def version_5(cls, node, **kwargs):
    return cls.make_node_from_tf_node(node, [node.inputs[0], node.inputs[1]])
