from onnx_tf.common import exception
from onnx_tf.handlers.frontend_handler import FrontendHandler


class TopK(FrontendHandler):
  _TF_OP = ["TopKV2"]

  @classmethod
  def param_check(cls, node, version, **kwargs):
    if node.inputs[1] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[1], node.op)

  @classmethod
  def version_1(cls, node, **kwargs):
    consts = kwargs["consts"]
    k = int(consts[node.inputs[1]])
    return cls.make_node(node, inputs=[node.inputs[0]], version=1, k=k, axis=-1)