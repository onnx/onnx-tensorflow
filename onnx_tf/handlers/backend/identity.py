import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.handlers.handler import partial_support
from onnx_tf.handlers.handler import ps_description

@onnx_op("Identity")
@tf_func(tf.identity)
@partial_support(True)
@ps_description("Identity with sequence type is not supported " +
                "in Tensorflow")
class Identity(BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]

  @classmethod
  def version_13(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]
