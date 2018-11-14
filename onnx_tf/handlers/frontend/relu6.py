from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import tf_op
from onnx_tf.common import get_unique_suffix


@tf_op("Relu6")
class Relu6(FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    output = "unclipped" + get_unique_suffix()
    nodes = [
        cls.make_node("Relu", node.inputs, [output], version=1),
        cls.make_node(
            "Clip", [output], node.outputs, min=0.0, max=6.0, version=1),
    ]
    return nodes

  @classmethod
  def version_6(cls, node, **kwargs):
    output = "unclipped" + get_unique_suffix()
    nodes = [
        cls.make_node("Relu", node.inputs, [output], version=1),
        cls.make_node(
            "Clip", [output], node.outputs, min=0.0, max=6.0, version=6),
    ]
    return nodes
