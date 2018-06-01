import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Sigmoid")
@tf_func(tf.nn.sigmoid)
class Sigmoid(BackendHandler):

  @classmethod
  def process_attrs(cls, attrs):
    return cls._process_attrs(attrs, remove=["consumed_inputs"])

  @classmethod
  def version_1(cls, node, **kwargs):
    return [cls.make_tf_tensor(node, **kwargs)]

  @classmethod
  def version_6(cls, node, **kwargs):
    return [cls.make_tf_tensor(node, **kwargs)]
