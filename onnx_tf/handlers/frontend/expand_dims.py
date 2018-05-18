from onnx_tf.common import exception
from onnx_tf.handlers.frontend_handler import FrontendHandler


class ExpandDims(FrontendHandler):
  _ONNX_OP = "Unsqueeze"

  @classmethod
  def param_check(cls, node, version, **kwargs):
    if node.inputs[1] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[1], node.op)

  @classmethod
  def version_1(cls, node, **kwargs):
    consts = kwargs["consts"]
    axes = [int(consts[node.inputs[1]])]
    return cls.make_node(node, [node.inputs[0]], version=1, axes=axes)
