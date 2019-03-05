import numpy as np
import onnx

from onnx_tf.common import exception
from onnx_tf.common.data_type import any_dtype_to_onnx_dtype
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("ConstantOfShape")
@tf_op("Fill")
class Fill(FrontendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    if node.inputs[1] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[1], node.op_type)

  @classmethod
  def version_1(cls, node, **kwargs):
    value = float(np.asscalar(kwargs["consts"][node.inputs[1]]))
    return cls.make_node_from_tf_node(
        node, [node.inputs[0]],
        op_type="ConstantFill",
        input_as_shape=1,
        value=value)

  @classmethod
  def version_9(cls, node, **kwargs):
    value_np_dtype = kwargs["consts"][node.inputs[1]].dtype
    value = np.asscalar(kwargs["consts"][node.inputs[1]])
    onnx_type = any_dtype_to_onnx_dtype(np_dtype=value_np_dtype)
    tensor_value = onnx.helper.make_tensor("value", onnx_type, [1], [value])
    return cls.make_node_from_tf_node(
        node, [node.inputs[0]], value=tensor_value)
