import copy

from onnx_tf.common import get_unique_suffix
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op
from .math_mixin import ArithmeticMixin


@onnx_op("Add")
@tf_op("BiasAdd")
class BiasAdd(ArithmeticMixin, FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    data_format = node.attr.get("data_format", "NHWC")
    channel_first = chr(data_format[1]) == "C" if isinstance(
        data_format[1], int) else data_format[1] == "C"
    axis = 1 if channel_first else -1
    return cls.arithmetic_op(node, axis=axis, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    data_format = node.attr.get("data_format", "NHWC")
    channel_first = chr(data_format[1]) == "C" if isinstance(
        data_format[1], int) else data_format[1] == "C"
    axis = 1 if channel_first else -1
    return cls.arithmetic_op(node, axis=axis, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    data_format = node.attr.get("data_format", "NHWC")
    channel_first = chr(data_format[1]) == "C" if isinstance(
        data_format[1], int) else data_format[1] == "C"
    axis = 1 if channel_first else -1

    unsqueeze_suffix = get_unique_suffix()
    if axis == 1:
      # In this case, we manually unsqueeze the bias term
      # to facilitate broadcasting.
      num_sp_dim = len(data_format) - 2
      unsqueeze_axes = [i + 1 for i in range(num_sp_dim)]
      reshape_node = cls.make_node_from_tf_node(
          node, [node.inputs[1]], [node.inputs[1] + "_" + unsqueeze_suffix],
          axes=unsqueeze_axes,
          op_type="Unsqueeze",
          name=node.inputs[1] + unsqueeze_suffix)
      node_update_input = copy.deepcopy(node)
      node_update_input.inputs = [
          node.inputs[0], node.inputs[1] + "_" + unsqueeze_suffix
      ]
      return [reshape_node, cls.arithmetic_op(node_update_input, **kwargs)]
    else:
      return cls.arithmetic_op(node, **kwargs)
