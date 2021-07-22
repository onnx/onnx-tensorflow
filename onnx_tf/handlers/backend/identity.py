import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Identity")
@tf_func(tf.identity)
class Identity(BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]

  @classmethod
  def version_13(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]

  @classmethod
  def version_14(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    if isinstance(x, (list, tuple)):
      return [tf.identity_n(x)]
    else:
      return [tf.identity(x)]
