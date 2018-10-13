import numpy as np

from onnx_tf.common import get_unique_suffix
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import tf_op


@tf_op("StridedSlice")
class StridedSlice(FrontendHandler):

  @classmethod
  def _int_to_set_pos_list(cls, num, num_bit=32):
    return np.where([bool(num & (1 << n)) for n in range(num_bit)])[0].tolist()

  @classmethod
  def version_1(cls, node, **kwargs):
    shrink_axis_mask = node.attr.get("shrink_axis_mask", 0)
    begin_mask = node.attr.get("begin_mask", 0)
    end_mask = node.attr.get("end_mask", 0)
    ellipsis_mask = node.attr.get("ellipsis_mask", 0)
    new_axis_mask = node.attr.get("new_axis_mask", 0)
    shrink_axis_mask = node.attr.get("shrink_axis_mask", 0)

    need_post_processing = (shrink_axis_mask > 0 or begin_mask > 0 or
                            end_mask > 0 or ellipsis_mask > 0 or
                            new_axis_mask > 0 or shrink_axis_mask > 0)

    slice_suffix = "_" + get_unique_suffix() if need_post_processing else ""
    slice_output_name = cls.get_outputs_names(node)[0]
    slice_node = cls.make_node("DynamicSlice", node.inputs[0:3],
                               [slice_output_name + slice_suffix],
                               node.name + slice_suffix)

    if not need_post_processing:
      return [slice_node]

    shrink_axis = cls._int_to_set_pos_list(shrink_axis_mask)
    squeeze_node = cls.make_node(
        "Squeeze", [slice_output_name + slice_suffix],
        cls.get_outputs_names(node),
        node.name,
        axes=shrink_axis)
    return [slice_node, squeeze_node]
