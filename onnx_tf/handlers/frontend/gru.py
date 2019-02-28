from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import experimental
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("GRU")
@tf_op("GRU")
@experimental
class GRU(FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.make_node_from_tf_node(node)

  @classmethod
  def version_3(cls, node, **kwargs):
    return cls.make_node_from_tf_node(node)

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls.make_node_from_tf_node(node)
