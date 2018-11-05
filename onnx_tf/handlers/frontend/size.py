import tensorflow as tf

from onnx.helper import make_node
from onnx_tf.common import get_unique_suffix
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op
from onnx_tf.handlers.frontend.cast import Cast


@onnx_op("Size")
@tf_op("Size")
class Size(FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    # tf.size out_type could be int32 or int64
    need_cast = node.attr['out_type'] == tf.int32

    size_suffix = "_" + get_unique_suffix() if need_cast else ""
    size_output_name = node.outputs[0] + size_suffix
    size_node = cls.make_node_from_tf_node(
        node, [node.inputs[0]],
        outputs=[size_output_name],
        name=node.name + size_suffix)

    if not need_cast:
      return [size_node]

    cast_node = Cast.handle(
        make_node(
            "Cast", [size_output_name],
            outputs=node.outputs,
            name=node.name,
            DstT=node.attr['out_type']))
    return [size_node, cast_node]
