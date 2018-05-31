import copy

import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.handlers.handler import tf_op


@onnx_op("ArgMax")
@tf_op("ArgMax")
@tf_func(tf.arg_max)
class ArgMax(BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    attrs = copy.deepcopy(node.attrs)
    attrs.setdefault("axis", 0)
    keepdims = attrs.pop("keepdims", 1)
    arg_max = cls.make_tf_tensor(node, attrs=attrs, **kwargs)
    if keepdims == 1:
      return [
        arg_max,
        cls.make_tf_tensor(node, tf_func=tf.expand_dims, inputs=[arg_max], attrs=attrs)
      ]
    return [arg_max]
