from onnx_tf.common import get_unique_suffix
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import tf_op


@tf_op("Pack")
class Pack(FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    axis = node.attr.get("axis", 0)
    unsqueeze_outputs = [
        i + "_Unsqueeze_" + get_unique_suffix() for i in node.inputs
    ]
    nodes = []
    for i, o in zip(node.inputs, unsqueeze_outputs):
      nodes.append(
          cls.make_node(
              "Unsqueeze", [i], [o],
              node.name + "_Unsqueeze_" + get_unique_suffix(),
              axes=[axis],
              version=1))
    concat = cls.make_node(
        "Concat",
        unsqueeze_outputs,
        node.outputs,
        node.name,
        axis=axis,
        version=4)
    return nodes + [concat]
