import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from .math_mixin import ReductionMixin


@onnx_op("ReduceLogSum")
class ReduceLogSum(ReductionMixin, BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    axis = node.attrs.get("axes", list(range(len(x.get_shape().as_list()))))
    keepdims = node.attrs.get("keepdims", 1) == 1
    return [tf.math.log(tf.reduce_sum(x, axis=axis, keepdims=keepdims))]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
