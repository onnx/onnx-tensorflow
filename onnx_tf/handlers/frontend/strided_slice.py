import numpy as np

from onnx_tf.common import exception
from onnx_tf.common import get_unique_suffix
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("DynamicSlice")
@tf_op("StridedSlice")
class StridedSlice(FrontendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    if node.inputs[3] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[3], node.op_type)

  # Convenience function to convert int bit mask to a list of bit indices
  # where the bit is set. For instance, _int_to_set_pos_list(5) -> [1, 3]
  # (since 5 has binary representatioin of 0101)
  @classmethod
  def _int_to_set_pos_list(cls, num, num_bit=32):
    return np.where([bool(num & (1 << n)) for n in range(num_bit)])[0].tolist()

  @classmethod
  def version_9(cls, node, **kwargs):
    begin_mask = node.attr.get("begin_mask", 0)
    end_mask = node.attr.get("end_mask", 0)
    ellipsis_mask = node.attr.get("ellipsis_mask", 0)
    new_axis_mask = node.attr.get("new_axis_mask", 0)
    shrink_axis_mask = node.attr.get("shrink_axis_mask", 0)

    only_support = (int(begin_mask) is 0 and int(end_mask) is 0 and
                    int(ellipsis_mask) is 0 and int(new_axis_mask) is 0)
    assert only_support, "limited strided slice support"

    # Assert that strides are all ones, since we have limited support.
    const_strides = kwargs["consts"][node.inputs[3]]
    np.testing.assert_array_equal(np.ones_like(const_strides), const_strides)

    need_post_processing = (shrink_axis_mask > 0 or begin_mask > 0 or
                            end_mask > 0 or ellipsis_mask > 0 or
                            new_axis_mask > 0 or shrink_axis_mask > 0)

    slice_suffix = "_" + get_unique_suffix() if need_post_processing else ""
    slice_output_name = node.outputs[0]
    slice_node = cls.make_node("DynamicSlice", node.inputs[0:3],
                               [slice_output_name + slice_suffix],
                               node.name + slice_suffix)

    if not need_post_processing:
      return [slice_node]

    shrink_axis = cls._int_to_set_pos_list(shrink_axis_mask)
    squeeze_node = cls.make_node(
        "Squeeze", [slice_output_name + slice_suffix],
        node.outputs,
        node.name,
        axes=shrink_axis)
    return [slice_node, squeeze_node]
