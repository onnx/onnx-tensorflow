import tensorflow as tf
import numpy as np

from onnx_tf.common import exception
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
    # Convert to NCHW
    transpose_node = Transpose.handle(
        TensorflowNode(
            name='Transpose',
            inputs=node.inputs[:1] + ["perm"],
            outputs=["transposed_input"]),
        consts={"perm": [0, 3, 1, 2]})

    input_shape_node = Shape.handle(
        TensorflowNode(
            name='Input_Shape',
            inputs=["transposed_input"],
            outputs=["input_shape"],
            attr=node.attr,
            op_type='Shape'))

    util_one = "_onnx_tf_internal_one_fp32"

    output_shape_tensor = node.inputs[1]

    out_shape_float = Cast.handle(
        TensorflowNode(
            name='Out_shape_Cast',
            inputs=[output_shape_tensor],
            outputs=["output_shape_float_partial"],
            attr={"DstT": tf.float32}))

    in_shape_cast = Cast.handle(
        TensorflowNode(
            name='In_shape_Cast',
            inputs=["input_shape"],
            outputs=["input_shape_float"],
            attr={"DstT": tf.float32}))

    in_shape_slice = Slice.handle(
        TensorflowNode(
            name="Slice",
            inputs=["input_shape_float", "begin", "size"],
            outputs=["sliced_input_shape"],
            attr={"axes": [0]}),
        consts={
            "begin": np.array([2]),
            "size": np.array([2])
        })

    div_node = Div.handle(
        TensorflowNode(
            name='Div',
            inputs=["output_shape_float_partial", "sliced_input_shape"],
            outputs=["hw_scale"]))

    full_scale = Concat.handle(
        TensorflowNode(
            name='Scale_Concat',
            inputs=[util_one, util_one, "hw_scale", "concat_axis"],
            outputs=["scale"]),
        consts={"concat_axis": 0})

    upsample_node = cls.make_node_from_tf_node(
        node,
        op_type="Upsample",
        mode="bilinear",
        inputs=["transposed_input", "scale"],
        outputs=["upsample_to_tranpose"])

    transpose_output_node = Transpose.handle(
        TensorflowNode(
            name='Transpose',
            inputs=["upsample_to_tranpose", "perm"],
            outputs=node.outputs),
        consts={"perm": [0, 2, 3, 1]})

    return [
        transpose_node, input_shape_node, out_shape_float, in_shape_cast,
        in_shape_slice, div_node, full_scale, upsample_node,
        transpose_output_node
    ]
