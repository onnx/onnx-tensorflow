from onnx import TensorProto

from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("Tile")
@tf_op("Tile")
class Tile(FrontendHandler):

  @classmethod
  def version_6(cls, node, **kwargs):
    data_type_cast_map = kwargs["data_type_cast_map"]
    data_type_cast_map[node.inputs[1]] = TensorProto.INT64
    return cls.make_node_from_tf_node(node)
