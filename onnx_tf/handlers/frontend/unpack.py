from onnx_tf.common import get_unique_suffix
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import tf_op


@tf_op("Unpack")
class Unpack(FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    axis = node.attr.get("axis", 0)
    outputs = node.outputs
    split_outputs = [o + "_Split_" + get_unique_suffix() for o in outputs]
    splited = cls.make_node(
        "Split",
        node.inputs,
        split_outputs,
        node.name,
        axis=axis,
        split=[1] * node.attr["num"],
        version=2)
    nodes = [splited]
    for split_output, output in zip(split_outputs, outputs):
      nodes.append(
          cls.make_node(
              "Squeeze", [split_output], [output],
              node.name + "_Squeeze_" + get_unique_suffix(),
              axes=[axis],
              version=1))
    return nodes
