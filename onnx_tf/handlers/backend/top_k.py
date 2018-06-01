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
    axes = list(range(x_rank))
    axis = node.attrs.get("axis", -1)
    axis = axis if axis >= 0 else axis + x_rank

    if axis != x_rank - 1:
      pre_perm = [a for a in axes if a != axis] + [axis]
      post_perm = axes[:axis] + [x_rank - 1] + axes[axis:x_rank - 1]
      x = tf.transpose(x, perm=pre_perm)
      values, indices = tf.nn.top_k(x, k=node.attrs["k"])
      values = tf.transpose(values, perm=post_perm)
      return [values, tf.cast(indices, dtype=tf.int64)]

    values, indices = tf.nn.top_k(x, k=node.attrs["k"])
    return [values, tf.cast(indices, dtype=tf.int64)]
