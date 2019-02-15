import numpy as np

from onnx.helper import make_node
from onnx.helper import make_tensor

from onnx_tf.common.data_type import any_dtype_to_onnx_dtype
from onnx_tf.common import get_unique_suffix
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import tf_op
from onnx_tf.handlers.frontend.pad import Pad
from onnx_tf.handlers.frontend.reshape import Reshape
from onnx_tf.handlers.frontend.transpose import Transpose
from onnx_tf.pb_wrapper import TensorflowNode


@tf_op("SpaceToBatchND")
class SpaceToBatchND(FrontendHandler):
  SINCE_VERSION = 9

  @classmethod
  def args_check(cls, node, **kwargs):
    if node.inputs[1] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[1], node.op_type)
    if node.inputs[2] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[2], node.op_type)

  @classmethod
  def version_9(cls, node, **kwargs):
    pad_suffix = '_' + get_unique_suffix()
    shape_suffix1 = '_' + get_unique_suffix()
    shape_suffix2 = '_' + get_unique_suffix()
    reshape_suffix1 = '_' + get_unique_suffix()
    reshape_suffix2 = '_' + get_unique_suffix()
    transpose_suffix = '_' + get_unique_suffix()

    pad_output_name = node.outputs[0] + pad_suffix
    shape_output_name1 = node.outputs[0] + shape_suffix1
    shape_output_name2 = node.outputs[0] + shape_suffix2
    reshape_output_name1 = node.outputs[0] + reshape_suffix1
    transpose_output_name = node.outputs[0] + transpose_suffix

    pad_name = 'pad' + pad_suffix
    reshape_name1 = 'reshape' + reshape_suffix1
    reshape_name2 = 'reshape' + reshape_suffix2
    transpose_name = 'transpose' + transpose_suffix

    # before pad, need to make sure the shape matches therefore
    # paddings must be shape of (n, 2) instead of (m, 2)
    block_shape = kwargs['consts'][node.inputs[1]]
    input_paddings = kwargs['consts'][node.inputs[2]]
    input_shape = kwargs['node_dict'][node.inputs[0]].attr['_output_shapes'][0]

    m = block_shape.shape[0]
    n = len(input_shape)

    padded_paddings = np.pad(input_paddings, ((1, n - m - 1), (0, 0)),
                             'constant')
    kwargs['consts']['padded_paddings'] = padded_paddings

    # step1: apply paddings
    pad_node = Pad.handle(
        TensorflowNode(
            name=pad_name,
            inputs=[node.inputs[0], 'padded_paddings'],
            outputs=[pad_output_name],
            attr=node.attr,
            domain=node.domain,
            op_type='Pad'), **kwargs)

    # step2: reshape
    first_input_shape = []
    first_input_shape.append(input_shape[0])
    for x, value in enumerate(block_shape):
      first_input_shape.append(
          int((input_shape[x + 1] + input_paddings[x][0] + input_paddings[x][1])
              / value))
      first_input_shape.append(value)
    for x in range(len(block_shape) + 1, len(input_shape)):
      first_input_shape.append(input_shape[x])
    input_shape_npa1 = np.asarray(first_input_shape, dtype=np.int32)

    shape_node1 = make_node(
        "Constant", [], [shape_output_name1],
        value=make_tensor(
            shape_output_name1,
            any_dtype_to_onnx_dtype(np_dtype=input_shape_npa1.dtype),
            input_shape_npa1.shape,
            input_shape_npa1))
    reshape_node1 = Reshape.handle(
        TensorflowNode(
            name=reshape_name1,
            inputs=[pad_output_name, shape_output_name1],
            outputs=[reshape_output_name1],
            attr=node.attr,
            domain=node.domain,
            op_type='Reshape'), **kwargs)

    # step3: permute dimensions
    perm = []
    for x, value in enumerate(block_shape):
      perm.append((x + 1) * 2)
    perm.append(0)
    for x, value in enumerate(block_shape):
      perm.append((x + 1) * 2 - 1)
    for x in range(len(perm), len(first_input_shape)):
      perm.append(x)

    transpose_node = Transpose.handle(
        TensorflowNode(
            name=transpose_name,
            inputs=[reshape_output_name1] + ["perm"],
            outputs=[transpose_output_name]),
        consts={
            "perm": perm
        })

    # step4: final reshape
    second_input_shape = []
    b = input_shape[0]
    for x in block_shape:
      b = b * x
    second_input_shape.append(b)
    for x, value in enumerate(block_shape):
      second_input_shape.append(
          int((input_shape[x + 1] + input_paddings[x][0] + input_paddings[x][1])
              / value))
    for x in range(len(block_shape) + 1, len(input_shape)):
      second_input_shape.append(input_shape[x])

    input_shape_npa2 = np.asarray(second_input_shape, dtype=np.int32)

    shape_node2 = make_node(
        "Constant", [], [shape_output_name2],
        value=make_tensor(
            shape_output_name2,
            any_dtype_to_onnx_dtype(np_dtype=input_shape_npa2.dtype),
            input_shape_npa2.shape,
            input_shape_npa2))

    reshape_node2 = Reshape.handle(
        TensorflowNode(
            name=reshape_name2,
            inputs=[transpose_output_name, shape_output_name2],
            outputs=node.outputs,
            attr=node.attr,
            domain=node.domain,
            op_type='Reshape'), **kwargs)

    return [
        pad_node, shape_node1, reshape_node1, transpose_node, shape_node2,
        reshape_node2
    ]
