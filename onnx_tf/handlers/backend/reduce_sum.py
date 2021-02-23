import copy

import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from .math_mixin import ReductionMixin
from onnx_tf.common.tf_helper import tf_shape


@onnx_op("ReduceSum")
@tf_func(tf.reduce_sum)
class ReduceSum(ReductionMixin, BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    attrs = copy.deepcopy(node.attrs)
    noop_with_empty_axes = attrs.pop("noop_with_empty_axes", 0)
    axis = None

    if len(node.inputs) > 1:
      axes = kwargs["tensor_dict"][node.inputs[1]]
      axes_shape = tf_shape(axes)
      if len(axes_shape) > 1:
        axis = axes
      else:
        axis = axes[0] if axes_shape[0] != 0 else axis

    # return the input tensor when axis is None and noop_with_empty_axes is True
    if axis is None and noop_with_empty_axes:
      return [x]

    attrs["axis"] = axis
    # https://github.com/onnx/onnx/issues/585
    attrs["keepdims"] = attrs.pop("keepdims", 1) == 1
    return [
        cls.make_tensor_from_onnx_node(node, inputs=[x], attrs=attrs, **kwargs)
    ]
