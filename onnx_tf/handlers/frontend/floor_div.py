from onnx_tf.common import get_unique_suffix
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import tf_op
from onnx_tf.handlers.frontend.div import Div
from onnx_tf.handlers.frontend.floor import Floor
from onnx_tf.pb_wrapper import TensorflowNode


@tf_op("FloorDiv")
class FloorDiv(FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    div_suffix = '_' + get_unique_suffix()
    div_output_name = node.outputs[0] + div_suffix
    div_node = Div.handle(
        TensorflowNode(
            name='Div',
            inputs=node.inputs[0:2],
            outputs=[div_output_name],
            attr=node.attr,
            domain=node.domain,
            op_type='Div'), **kwargs)
    floor_node = Floor.handle(
        TensorflowNode(
            name='Floor',
            inputs=[div_output_name],
            outputs=node.outputs,
            attr=node.attr,
            domain=node.domain,
            op_type='Floor'))
    return [div_node, floor_node]
