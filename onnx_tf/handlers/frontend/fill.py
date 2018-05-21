import numpy as np

from onnx_tf.common import exception
from onnx_tf.handlers.frontend_handler import FrontendHandler


class Fill(FrontendHandler):
  ONNX_OP = "ConstantFill"

  @classmethod
  def param_check(cls, node, version, **kwargs):
    if node.inputs[1] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[1], node.op)

  @classmethod
  def version_1(cls, node, **kwargs):
    value = float(np.asscalar(kwargs["consts"][node.inputs[1]]))
    return cls.make_node(
        node, [node.inputs[0]], input_as_shape=1, value=value)
