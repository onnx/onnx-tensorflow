import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Hardmax")
@tf_func(tfa.seq2seq.hardmax)
class Hardmax(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]

    if cls.SINCE_VERSION < 13:
      axis = node.attrs.get("axis", 1)
      axis = axis if axis >= 0 else len(np.shape(x)) + axis

      if axis == len(np.shape(x)) - 1:
        return [cls.make_tensor_from_onnx_node(node, **kwargs)]

      shape = tf.shape(x)
      cal_shape = (tf.reduce_prod(shape[0:axis]),
                   tf.reduce_prod(shape[axis:tf.size(shape)]))
      x = tf.reshape(x, cal_shape)
      return [tf.reshape(tfa.seq2seq.hardmax(x), shape)]

    else: # opset 13
      axis = node.attrs.get("axis", -1) # default for axis is -1 in opset 13
      axis = axis if axis >= 0 else len(np.shape(x)) + axis

      if axis == len(np.shape(x)) - 1:
        return [cls.make_tensor_from_onnx_node(node, **kwargs)]

      perm1 = tf.range(0, axis)
      perm2 = tf.range(axis + 1, len(tf.shape(x)) - 1)
      perm = tf.concat([perm1, [len(tf.shape(x)) - 1], perm2, [axis]], -1)
      x = tf.transpose(x, perm)

      return [tf.transpose(tfa.seq2seq.hardmax(x), perm)]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
