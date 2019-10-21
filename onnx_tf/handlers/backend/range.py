import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Range")
@tf_func(tf.range)
class Round(BackendHandler):

  @classmethod
  def version_11(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]
