from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("Constant")
@tf_op("ZerosLike")
class ZerosLike(FrontendHandler):

  @classmethod
  def version_9(cls, node, **kwargs):
    consts = kwargs["consts"]
    value = np.zeros_like(consts[node.inputs[0]])
    return cls.make_node_from_tf_node(node, value=value)  
