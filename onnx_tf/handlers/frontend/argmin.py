import numpy as np

from onnx_tf.common import exception
from onnx_tf.handlers.frontend_handler import FrontendHandler


class ArgMin(FrontendHandler):

  @classmethod
  def param_check(cls, node, version, **kwargs):
    if node.inputs[1] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[1], node.op)

  @classmethod
  def version_1(cls, node, **kwargs):
    axis = np.asscalar(kwargs["consts"][node.inputs[1]])
    return cls.make_node(
        node, [node.inputs[0]], version=1, axis=axis, keepdims=0)
