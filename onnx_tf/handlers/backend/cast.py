import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.handlers.handler import tf_op


@onnx_op("Cast")
@tf_op("Cast")
@tf_func(tf.cast)
class Cast(BackendHandler):

  @classmethod
  def process_attrs(cls, attrs):
    attrs["dtype"] = attrs.pop("to")
    return attrs

  @classmethod
  def version_1(cls, node, **kwargs):
    return [cls.make_tf_tensor(node, **kwargs)]

  @classmethod
  def version_6(cls, node, **kwargs):
    return [cls.make_tf_tensor(node, **kwargs)]
