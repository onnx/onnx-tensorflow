import copy

import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Dropout")
@tf_func(tf.nn.dropout)
class Dropout(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    attrs = copy.deepcopy(node.attrs)
    if cls.SINCE_VERSION >= 7 or attrs.pop("is_test", 0) == 1:
      return [x]
    attrs["keep_prob"] = 1 - attrs.pop("ratio", 0.5)
    return [cls.make_tensor_from_onnx_node(node, attrs=attrs, **kwargs)]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_10(cls, node, **kwargs):
    return cls._common(node, **kwargs)
