import tensorflow as tf
import numpy as np

from onnx.helper import make_node
from onnx.helper import make_tensor

from onnx_tf.common import get_unique_suffix
from onnx_tf.common.data_type import any_dtype_to_onnx_dtype
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.frontend.shape import Shape
from onnx_tf.handlers.frontend.strided_slice import StridedSlice
from onnx_tf.handlers.frontend.cast import Cast
from onnx_tf.handlers.frontend.concat import Concat
from onnx_tf.handlers.frontend.div import Div
from onnx_tf.handlers.frontend.transpose import Transpose
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op
from onnx_tf.pb_wrapper import TensorflowNode
from onnx_tf.pb_wrapper import OnnxGraph


@onnx_op("Upsample")
@tf_op("ResizeBilinear")
class ResizeBilinear(FrontendHandler):

  @classmethod
  def version_9(cls, node, **kwargs):
    unique_suffix = get_unique_suffix()

    # Convert to NCHW:
    transpose_node = Transpose.handle(
        TensorflowNode(
            name='transopose_input_to_nchw_' + unique_suffix,
            inputs=node.inputs[:1] + ["perm"],
            outputs=["transposed_input_" + unique_suffix]),
        consts={"perm": [0, 3, 1, 2]})

    # Get shape of NCHW input tensor:
    input_shape_node = Shape.handle(
        TensorflowNode(
            name='get_input_shape_' + unique_suffix,
            inputs=transpose_node.output,
            outputs=["input_shape_" + unique_suffix],
            attr=node.attr))

    util_one = OnnxGraph.CONST_ONE_FP32

    output_shape_tensor = node.inputs[1]

    # Cast output shape (HW dim only) to float32:
    out_shape_float = Cast.handle(
        TensorflowNode(
            name='cast_output_shape_to_fp32_' + unique_suffix,
            inputs=[output_shape_tensor],
            outputs=["output_shape_float_partial_" + unique_suffix],
            attr={"DstT": tf.float32}))

    # Cast input shape to float32:
    in_shape_cast = Cast.handle(
        TensorflowNode(
            name='cast_input_shape_to_fp32_' + unique_suffix,
            inputs=input_shape_node.output,
            outputs=["input_shape_float_" + unique_suffix],
            attr={"DstT": tf.float32}))

    slice_const_items = [
        ("begin", np.array([2]).astype(np.int32)),
        ("end", np.array([4]).astype(np.int32)),
        ("strides", np.array([1]).astype(np.int32)),
    ]

    slice_const_proto = {}

    for k, v in slice_const_items:
      const_name = "{}_".format(k) + unique_suffix
      slice_const_proto[k] = make_node(
          "Constant", [], [const_name],
          value=make_tensor(
              const_name, any_dtype_to_onnx_dtype(np_dtype=v.dtype), v.shape,
              v))

    in_shape_slices = StridedSlice.handle(
        TensorflowNode(
            name="stridedslice_input_shape_" + unique_suffix,
            inputs=list(in_shape_cast.output) +
            [slice_const_proto[k].output[0] for k, v in slice_const_items],
            outputs=["sliced_input_shape_" + unique_suffix]),
        consts={
            slice_const_proto[k].output[0]: v for k, v in slice_const_items
        },
        add_consts=True)

    # Divide input shape with output shape to get scaling factor:
    div_node = Div.handle(
        TensorflowNode(
            name='div_to_get_scale_factor_' + unique_suffix,
            inputs=list(out_shape_float.output) + list(
                in_shape_slices[-1].output),
            outputs=["hw_scale_" + unique_suffix]))

    # Prepend 1's in the N, C dimension:
    full_scale = Concat.handle(
        TensorflowNode(
            name='prepend_ones_to_scale_factor_' + unique_suffix,
            inputs=[util_one, util_one] + list(div_node.output) +
            ["concat_axis"],
            outputs=["scale_" + unique_suffix]),
        consts={"concat_axis": 0})

    # Upsample with the computed scaling factor:
    upsample_node = cls.make_node_from_tf_node(
        node,
        op_type="Upsample",
        mode="bilinear",
        inputs=list(transpose_node.output) + list(full_scale.output),
        outputs=["upsample_to_tranpose_" + unique_suffix])

    # Transpose back to NHWC:
    transpose_output_node = Transpose.handle(
        TensorflowNode(
            name='transpose_output_to_nhwc_' + unique_suffix,
            inputs=list(upsample_node.output) + ["perm"],
            outputs=node.outputs),
        consts={"perm": [0, 2, 3, 1]})

    transpose_and_get_shapes = [
        transpose_node, input_shape_node, out_shape_float, in_shape_cast
    ]
    slice_shape = list(slice_const_proto.values()) + in_shape_slices
    get_scale_and_upsample_and_transpose = [
        div_node, full_scale, upsample_node, transpose_output_node
    ]

    return transpose_and_get_shapes + slice_shape + get_scale_and_upsample_and_transpose
