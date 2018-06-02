from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("Squeeze")
@tf_op("Squeeze")
class Squeeze(FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    node_dict = kwargs["node_dict"]
    axes = node.attr.get("axis")
    if not axes:
      shape = node_dict[node.inputs[0]].attr["_output_shapes"][0]
      axes = [i for i, x in enumerate(shape) if x == 1]
    return cls.make_node_from_tf_node(node, [node.inputs[0]], axes=axes)
