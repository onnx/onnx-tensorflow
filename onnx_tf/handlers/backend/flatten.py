import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Flatten")
@tf_func(tf.keras.backend.flatten)
class Flatten(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    shape = tf.shape(x)
    axis = node.attrs.get("axis", 1)

    if axis == 0:
      cal_shape = (1, -1)
    else:
      cal_shape = (tf.reduce_prod(shape[0:axis]),
                   tf.reduce_prod(shape[axis:tf.size(shape)]))
    return [tf.reshape(x, cal_shape)]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_9(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
