import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Squeeze")
@tf_func(tf.squeeze)
class Squeeze(BackendHandler):

  @classmethod
  def get_attrs_processor_param(cls):
    return {"rename": {"axes": "axis"}}

  @classmethod
  def version_1(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]

  @classmethod
  def version_11(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]

  @classmethod
  def version_13(cls, node, **kwargs):
    def reshape(x_shape, axes, rank):
      for ax in axes:
        if tf.less(ax, rank):
          tf.autograph.experimental.set_loop_options(
          shape_invariants=[(x_shape, tf.TensorShape([None]))])
          x_shape = tf.concat([x_shape[:ax], x_shape[ax+1:]], axis=0)
      return x_shape

    axes = kwargs["tensor_dict"][node.inputs[1]]
    axes = tf.sort(axes)
    x = kwargs["tensor_dict"][node.inputs[0]]

    if axes.shape != 0:
      x_shape = tf.shape(x)
      rank = tf.cast(tf.rank(x), tf.int64)
      reshape_x = reshape(x_shape, axes, rank)
      x = tf.reshape(x, reshape_x)

    return [x]
