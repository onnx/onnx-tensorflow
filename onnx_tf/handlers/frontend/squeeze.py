from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.frontend_handler import version


class Squeeze(FrontendHandler):

  @classmethod
  @version(1)
  def version_1(cls, node, **kwargs):
    node_dict = kwargs["node_dict"]
    axes = node.attr.get("axis")
    if not axes:
      shape = node_dict[node.inputs[0]].attr["_output_shapes"][0]
      axes = [i for i, x in enumerate(shape) if x == 1]
    return cls.make_node(node, [node.inputs[0]], axes=axes)
