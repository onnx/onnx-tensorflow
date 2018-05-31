import numpy as np

from onnx_tf.common import exception
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("Pad")
@tf_op("Pad")
class Pad(FrontendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    if node.inputs[1] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[1], node.op_type)
    supported_modes = ["constant", "reflect"]
    mode = node.attr.get("mode", "constant")
    if mode.lower() not in supported_modes:
      raise RuntimeError(
          "Mode {} of Pad is not supported in ONNX.".format(mode))

  @classmethod
  def version_1(cls, node, **kwargs):
    consts = kwargs["consts"]
    mode = node.attr.get("mode", "constant")
    pads = np.transpose(consts[node.inputs[1]]).flatten()
    return cls.make_node_from_tf_node(
        node, [node.inputs[0]], paddings=pads, mode=mode, value=0.0)

  @classmethod
  def version_2(cls, node, **kwargs):
    consts = kwargs["consts"]
    mode = node.attr.get("mode", "constant")
    pads = np.transpose(consts[node.inputs[1]]).flatten()
    return cls.make_node_from_tf_node(
        node, [node.inputs[0]], pads=pads, mode=mode, value=0.0)
