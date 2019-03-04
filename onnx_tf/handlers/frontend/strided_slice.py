import numpy as np
import tensorflow as tf

from onnx_tf.common import exception
from onnx_tf.common import get_unique_suffix
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.frontend.cast import Cast
from onnx_tf.handlers.handler import experimental
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op
from onnx_tf.pb_wrapper import TensorflowNode

@onnx_op("DynamicSlice")
@tf_op("StridedSlice")
@experimental
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
  def version_1(cls, node, **kwargs):
    begin_mask = node.attr.get("begin_mask", 0)
    end_mask = node.attr.get("end_mask", 0)
    ellipsis_mask = node.attr.get("ellipsis_mask", 0)
    new_axis_mask = node.attr.get("new_axis_mask", 0)
    shrink_axis_mask = node.attr.get("shrink_axis_mask", 0)

    only_support = (int(ellipsis_mask) is 0 and int(new_axis_mask) is 0)

    preprocessing_node = []

    begin_node_name = node.inputs[1]
    end_node_name = node.inputs[2]

    # Process to begin or end mask.
    def process_range_mask(mask, range_array, begin_or_end_str,
                           preprocessing_node):
      """
        Args:
            mask (int): The begin or end mask
            range_array (str): Onnx tensor name corresponding to the
              begin or end tensor.
            begin_or_end_str (str): "begin" or "end".
            preprocessing_node (list): Array corresponding to the list
              of preprocessing nodes.
        Returns:
            str: returns Onnx tensor name corresponding to the processed
            begin or end tensor. In case where no processing is needed,
            the casted (to int64) begin or end tensor is returned.
      """

      # If there's no mask, we cast range array to int64 and return.
      range_cast_node_name = "{}_{}_range_casted".format(
          node.name, begin_or_end_str)
      range_cast_node = Cast.handle(
          TensorflowNode(
              name=range_cast_node_name,
              inputs=[range_array],
              outputs=[range_cast_node_name],
              attr={"DstT": tf.int64}))
      preprocessing_node.append(range_cast_node)

      if mask == 0 or mask is None:
        return range_cast_node_name

      # Indices and values for scatter to upadate begin or end array.
      # For instance if begin array is [1, 2, 3] and set axes in mask is
      # [1, 2], then we need to update begin array to [1, 0, 0].
      # To do so we construct Scatter operator with:
      # - data = [1, 2, 3]
      # - indices = [[1, 2]]
      # - updates = [0, 0]
      indices_name = "{}_{}_indices".format(node.name, begin_or_end_str)
      values_name = "{}_{}_values".format(node.name, begin_or_end_str)
      kwargs["additional_constants"][indices_name] = np.array(
          map(lambda x: [x], cls._int_to_set_pos_list(mask))).astype(np.int64)

      # Value to replace is 0 for begin array (slice from the very beginning of
      # this dimension) and -1 for end array (slice till the very end of
      # this dimension)
      value = 0 if begin_or_end_str == "begin" else -1
      kwargs["additional_constants"][values_name] = np.array(
            [value for i in cls._int_to_set_pos_list(mask)]).astype(np.int64)

      # Create processed(masked) range array.
      range_masked_node_name = "{}_{}_masked".format(node.name,
                                                     begin_or_end_str)
      range_masked_node = cls.make_node(
          "Scatter", [range_cast_node_name, indices_name, values_name],
          [range_masked_node_name], range_masked_node_name)

      # Include cast and scatter node as preprocessing nodes.
      preprocessing_node.append(range_masked_node)
      return range_masked_node_name

    begin_node_name = process_range_mask(begin_mask, begin_node_name, "begin",
                                         preprocessing_node)
    end_node_name = process_range_mask(end_mask, end_node_name, "end",
                                       preprocessing_node)

    assert only_support, "limited strided slice support"

    # Assert that strides are all ones, since we have limited support.
    const_strides = kwargs["consts"][node.inputs[3]]
    np.testing.assert_array_equal(np.ones_like(const_strides), const_strides)

    need_post_processing = (ellipsis_mask > 0 or new_axis_mask > 0 or
                            shrink_axis_mask > 0)

    slice_suffix = "_" + get_unique_suffix() if need_post_processing else ""
    slice_output_name = node.outputs[0]
    slice_node = cls.make_node(
        "DynamicSlice", [node.inputs[0], begin_node_name, end_node_name],
        [slice_output_name + slice_suffix], node.name + slice_suffix)

    if not need_post_processing:
      return preprocessing_node + [slice_node]

    shrink_axis = cls._int_to_set_pos_list(shrink_axis_mask)
    squeeze_node = cls.make_node(
        "Squeeze", [slice_output_name + slice_suffix],
        node.outputs,
        node.name,
        axes=shrink_axis)

    return preprocessing_node + [slice_node, squeeze_node]
