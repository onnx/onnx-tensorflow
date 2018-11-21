import tensorflow as tf

from onnx_tf.common import get_unique_suffix
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op
from onnx_tf.handlers.frontend.cast import Cast
from onnx_tf.pb_wrapper import TensorflowNode


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

    attrs = {}
    attrs['DstT'] = node.attr['out_type']

    cast_node = Cast.handle(
        TensorflowNode(
            name=node.name,
            inputs=[size_output_name],
            outputs=node.outputs,
            op_type='Cast',
            attr=attrs))
    return [size_node, cast_node]
