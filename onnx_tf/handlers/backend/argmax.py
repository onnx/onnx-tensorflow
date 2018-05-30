import copy

import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.handlers.handler import tf_op
from .math_mixin import ArithmeticMixin


@onnx_op("ArgMax")
@tf_op("ArgMax")
@tf_func(tf.arg_max)
class ArgMax(ArithmeticMixin, BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    attrs = copy.deepcopy(node.attrs)
    axis = attrs.pop("axis", 0)
    keepdims = attrs.get("keepdims", 1)
    if keepdims == 0:
      return [cls.make_tf_node(node, attrs=attrs, **kwargs)]
    else:
      arg_max_node = cls.make_tf_node(node, **kwargs)
      return [
        arg_max_node,
        cls.make_tf_node(node, tf_func=tf.expand_dims, attrs=attrs)
      ]

