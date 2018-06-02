import numpy as np
import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Hardmax")
@tf_func(tf.contrib.seq2seq.hardmax)
class Hardmax(BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    axis = node.attrs.get("axis", 1)
    axis = axis if axis >= 0 else len(np.shape(x)) + axis

    if axis == len(np.shape(x)) - 1:
      return [cls.make_tensor_from_onnx_node(node, **kwargs)]

    shape = tf.shape(x)
    cal_shape = (tf.reduce_prod(shape[0:axis]),
                 tf.reduce_prod(shape[axis:tf.size(shape)]))
    x = tf.reshape(x, cal_shape)

    return [tf.reshape(tf.contrib.seq2seq.hardmax(x), shape)]
