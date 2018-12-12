from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import tf_op
from onnx_tf.handlers.frontend.multiply import Multiply
from onnx_tf.pb_wrapper import TensorflowNode


@tf_op("Square")
class Square(FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    mul_node = Multiply.handle(
        TensorflowNode(
            name='Mul',
            inputs=[node.inputs[0], node.inputs[0]],
            outputs=node.outputs,
            attr=node.attr,
            domain=node.domain,
            op_type='Mul'), **kwargs)
    return mul_node
