import tensorflow as tf
import numpy as np

from onnx_tf.common import exception
from onnx_tf.common import get_unique_suffix
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.frontend.shape import Shape
from onnx_tf.handlers.frontend.slice import Slice
from onnx_tf.handlers.frontend.cast import Cast
from onnx_tf.handlers.frontend.concat import Concat
from onnx_tf.handlers.frontend.div import Div
from onnx_tf.handlers.frontend.fill import Fill
from onnx_tf.handlers.frontend.transpose import Transpose
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op
from onnx_tf.pb_wrapper import TensorflowNode


@onnx_op("Upsample")
@tf_op("ResizeBilinear")
class ResizeBilinear(FrontendHandler):

  @classmethod
  def version_9(cls, node, **kwargs):
    unique_suffix = get_unique_suffix()

    # Convert to NCHW:
    transpose_node = Transpose.handle(
        TensorflowNode(
            name='Transpose_In_' + unique_suffix,
            inputs=node.inputs[:1] + ["perm"],
            outputs=["transposed_input" + unique_suffix]),
        consts={"perm": [0, 3, 1, 2]})

    # Get shape of NCHW input tensor:
    input_shape_node = Shape.handle(
        TensorflowNode(
            name='Input_Shape',
            inputs=["transposed_input" + unique_suffix],
            outputs=["input_shape"],
            attr=node.attr,
            op_type='Shape'))

    util_one = "_onnx_tf_internal_one_fp32"

    output_shape_tensor = node.inputs[1]

    # Cast output shape (HW dim only) to float32:
    out_shape_float = Cast.handle(
        TensorflowNode(
            name='Out_shape_Cast',
            inputs=[output_shape_tensor],
            outputs=["output_shape_float_partial_" + unique_suffix],
            attr={"DstT": tf.float32}))

    # Cast input shape to float32:
    in_shape_cast = Cast.handle(
        TensorflowNode(
            name='In_shape_Cast',
            inputs=["input_shape"],
            outputs=["input_shape_float_" + unique_suffix],
            attr={"DstT": tf.float32}))

    # Get input shape in spatial dimensions only.
    in_shape_slice = Slice.handle(
        TensorflowNode(
            name="Slice",
            inputs=["input_shape_float_" + unique_suffix, "begin", "size"],
            outputs=["sliced_input_shape_" + unique_suffix],
            attr={"axes": [0]}),
        consts={
            "begin": np.array([2]),
            "size": np.array([2])
        })

    # Divide input shape with output shape to get scaling factor:
    div_node = Div.handle(
        TensorflowNode(
            name='Div',
            inputs=[
                "output_shape_float_partial_" + unique_suffix,
                "sliced_input_shape_" + unique_suffix
            ],
            outputs=["hw_scale_" + unique_suffix]))

    # Prepend 1's in the N, C dimension:
    full_scale = Concat.handle(
        TensorflowNode(
            name='Scale_Concat',
            inputs=[
                util_one, util_one, "hw_scale_" + unique_suffix, "concat_axis"
            ],
            outputs=["scale_" + unique_suffix]),
        consts={"concat_axis": 0})

    # Upsample with the computed scaling factor:
    upsample_node = cls.make_node_from_tf_node(
        node,
        op_type="Upsample",
        mode="bilinear",
        inputs=["transposed_input" + unique_suffix, "scale_" + unique_suffix],
        outputs=["upsample_to_tranpose" + unique_suffix])

    # Transpose back to NHWC:
    transpose_output_node = Transpose.handle(
        TensorflowNode(
            name='Transpose',
            inputs=["upsample_to_tranpose" + unique_suffix, "perm"],
            outputs=node.outputs),
        consts={"perm": [0, 2, 3, 1]})

    return [
        transpose_node, input_shape_node, out_shape_float, in_shape_cast,
        in_shape_slice, div_node, full_scale, upsample_node,
        transpose_output_node
    ]
