import copy
import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Compress")
@tf_func(tf.gather)
class Compress(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    attrs = copy.deepcopy(node.attrs)
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]
    condition = tensor_dict[node.inputs[1]]

    x = tf.reshape(x, [-1]) if node.attrs.get("axis") is None else x
    if condition.shape.is_fully_defined():
      condition_shape = condition.shape[0]
      indices = tf.constant(list(range(condition_shape)), dtype=tf.int64)
    else:
      condition_shape = tf.shape(condition, out_type=tf.int64)[0]
      indices = tf.range(condition_shape, dtype=tf.int64)
    not_zero = tf.not_equal(condition, tf.zeros_like(condition))
    attrs['indices'] = tf.boolean_mask(indices, not_zero)
    return [
        cls.make_tensor_from_onnx_node(node, inputs=[x], attrs=attrs, **kwargs)
    ]

  @classmethod
  def version_9(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)
