from onnx_tf.common import exception
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.frontend_handler import version


class Reshape(FrontendHandler):

  @classmethod
  def param_check(cls, node, **kwargs):
    if cls.SINCE_VERSION == 1:
      if node.inputs[1] not in kwargs["consts"]:
        exception.CONST_NOT_FOUND_EXCEPT(node.inputs[1], node.op)

  @classmethod
  @version(1)
  def version_1(cls, node, **kwargs):
    consts = kwargs["consts"]
    shape = consts[node.inputs[1]]
    return cls.make_node(node, [node.inputs[0]], shape=shape)

  @classmethod
  @version(5)
  def version_5(cls, node, **kwargs):
    return cls.make_node(node, [node.inputs[0], node.inputs[1]])
