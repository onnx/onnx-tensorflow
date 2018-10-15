from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op
from onnx_tf.common import get_unique_suffix

import tensorflow as tf
from onnx import TensorProto


@onnx_op("Shape")
@tf_op("Shape")
class Shape(FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    out_type = node.attr.get("out_type", tf.int32)

    # A flag to indicate whether output is int32.
    # If so, we need to insert a Cast node because
    # ONNX shape op only supports int64 as output type.
    need_cast_to_int32 = False
    if tf.as_dtype(out_type) == tf.int32:
      need_cast_to_int32 = True

    shape_suffix = "_" + get_unique_suffix() if need_cast_to_int32 else ""
    shape_name = cls.get_outputs_names(node)[0] + shape_suffix
    shape_node = cls.make_node_from_tf_node(
        node, outputs=[shape_name], name=shape_name)

    if need_cast_to_int32:
      dst_t = TensorProto.INT32
      cast_node = cls.make_node_from_tf_node(
          node,
          inputs=[shape_name],
          outputs=cls.get_outputs_names(node),
          op_type="Cast",
          to=TensorProto.INT32)
      return [shape_node, cast_node]

    return [shape_node]
