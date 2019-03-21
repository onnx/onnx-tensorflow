from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("Where")
@tf_op("Select")
class Select(FrontendHandler):

  @classmethod
  def version_9(cls, node, **kwargs):
    return cls.make_node_from_tf_node(node)
