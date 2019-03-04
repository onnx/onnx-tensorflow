import numpy as np

from onnx_tf.common import get_unique_suffix
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.frontend.shape import Shape
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("Constant")
@tf_op("ZerosLike")
class ZerosLike(FrontendHandler):

  @classmethod
  def version_9(cls, node, **kwargs):
    '''
      Currently, zeros_like output is an array with float32 type.
      However, it really needs to be with the same type as the input
      tensor. This is however currently not possible because
      we can't (always) get the type information of input tensor.
    '''
    input_tensor_shape = "input_shape_" + get_unique_suffix()
    input_tensor = node.inputs[0]

    input_tensor_shape_node = Shape.handle(
          TensorflowNode(
              name=input_tensor_shape,
              inputs=[range_array],
              outputs=[input_tensor_shape]))

    zeros_name = "zeros_" + get_unique_suffix()
    zeros_node = cls.make_node(
        "ConstantOfShape", [input_tensor_shape],
        [zeros_name], zeros_name)

    return [input_tensor_shape_node, zeros_node]
