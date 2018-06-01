import copy

import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Unsqueeze")
@tf_func(tf.expand_dims)
class Unsqueeze(BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    attrs = copy.deepcopy(node.attrs)
    axes = attrs.pop("axes")
    if len(axes) != 1:
      x = kwargs["tensor_dict"][node.inputs[0]]
      for axis in axes:
        x = tf.expand_dims(x, axis=axis, **kwargs)
      return [x]
    attrs["axis"] = axes[0]
    return [cls.make_tf_tensor(node, attrs=attrs, **kwargs)]
