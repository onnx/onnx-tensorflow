import tensorflow as tf

from onnx_tf.common.tf_helper import tf_shape
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("IsInf")
@tf_func(tf.math.is_inf)
class IsInf(BackendHandler):

  @classmethod
  def version_10(cls, node, **kwargs):
    inp = kwargs["tensor_dict"][node.inputs[0]]
    dtype = inp.dtype
    shape = tf_shape(inp)
    zero = tf.zeros(shape, dtype)
    dn = node.attrs.get("detect_negative", 1)
    dp = node.attrs.get("detect_positive", 1)
    # detecting only positive infinity, zero out elements < 0
    if dn == 0:
      inp = tf.maximum(zero, inp)
    # detecting only negative infinity, zero out elements > 0
    if dp == 0:
      inp = tf.minimum(zero, inp)
    return [cls.make_tensor_from_onnx_node(node, inputs=[inp], **kwargs)]
