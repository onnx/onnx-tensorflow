import copy
import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from .gather_and_scatter_mixin import GatherAndScatterMixin


@onnx_op("Gather")
@tf_func(tf.gather)
class Gather(GatherAndScatterMixin, BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    indices = kwargs["tensor_dict"][node.inputs[1]]
    attrs = copy.deepcopy(node.attrs)
    axis = attrs.get("axis", 0)
    result = cls.chk_idx_out_of_bounds_along_axis(x, axis, indices)
    msg = 'Gather indices are out of bounds, please double check the indices and retry.'
    with tf.control_dependencies([tf.compat.v1.assert_equal(result, True, message=msg)]):
      indices = cls.process_neg_idx_along_axis(x, axis, indices)
      attrs['axis'] = axis
      return [cls.make_tensor_from_onnx_node(node, attrs=attrs, inputs=[x, indices], **kwargs)]

  @classmethod
  def version_1(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
