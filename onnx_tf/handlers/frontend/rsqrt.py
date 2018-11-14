from onnx_tf.common import get_unique_suffix
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import tf_op
from onnx_tf.handlers.frontend.sqrt import Sqrt
from onnx_tf.handlers.frontend.reciprocal import Reciprocal
from onnx_tf.pb_wrapper import TensorflowNode


@tf_op("Rsqrt")
class Rsqrt(FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    rsqrt_suffix = "_" + get_unique_suffix()
    rsqrt_output_name = cls.get_outputs_names(node)[0] + rsqrt_suffix

    sqrt_node = Sqrt.handle(
        TensorflowNode(
            op_type='Sqrt',
            name=node.name + rsqrt_suffix,
            inputs=[node.inputs[0]],
            outputs=[rsqrt_output_name],
            attr=node.attr), **kwargs)

    reciprocal_node = Reciprocal.handle(
        TensorflowNode(
            op_type='Reciprocal',
            inputs=[rsqrt_output_name],
            outputs=cls.get_outputs_names(node),
            name=node.name,
            attr=node.attr), **kwargs)
    return [sqrt_node, reciprocal_node]
