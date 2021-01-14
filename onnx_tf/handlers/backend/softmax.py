import numpy as np
import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Softmax")
@tf_func(tf.nn.softmax)
class Softmax(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    if cls.SINCE_VERSION < 13:
      x = kwargs["tensor_dict"][node.inputs[0]]
      axis = node.attrs.get("axis", 1)
      axis = axis if axis >= 0 else len(np.shape(x)) + axis

      if axis == len(np.shape(x)) - 1:
        return [cls.make_tensor_from_onnx_node(node, **kwargs)]

      shape = tf.shape(x)
      cal_shape = (tf.reduce_prod(shape[0:axis]),
                   tf.reduce_prod(shape[axis:tf.size(shape)]))
      x = tf.reshape(x, cal_shape)

      return [tf.reshape(tf.nn.softmax(x), shape)]

    else:
      return [cls.make_tensor_from_onnx_node(node, **kwargs)]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
