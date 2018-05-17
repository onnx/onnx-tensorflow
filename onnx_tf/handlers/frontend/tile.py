from onnx import TensorProto

from onnx_tf.handlers.frontend_handler import FrontendHandler


class Tile(FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    data_type_cast_map = kwargs["data_type_cast_map"]
    data_type_cast_map[node.inputs[1]] = TensorProto.INT64
    return cls.make_node(node, version=1)
