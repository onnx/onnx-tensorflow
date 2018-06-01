import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("TopK")
@tf_func(tf.nn.top_k)
class TopK(BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    x_rank = len(x.get_shape())
    axis = node.attrs.get("axis", -1)
    if axis != -1 or axis != x_rank - 1:
      pass
    values, indices = tf.nn.top_k(x, k=node.attrs["k"])
    return [values, tf.cast(indices, dtype=tf.int64)]
