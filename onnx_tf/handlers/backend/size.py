import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Size")
@tf_func(tf.size)
class Size(BackendHandler):

  @classmethod
  def process_attrs(cls, attrs):
    return cls._process_attrs(attrs, default={"out_type": tf.int64})

  @classmethod
  def version_1(cls, node, **kwargs):
    return [cls.make_tf_tensor(node, **kwargs)]
