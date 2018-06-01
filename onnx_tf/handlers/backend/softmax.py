import numpy as np
import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Softmax")
@tf_func(tf.nn.softmax)
class Softmax(BackendHandler):

  @classmethod
  def process_attrs(cls, attrs):
    return cls._process_attrs(attrs, remove=["axis"])

  @classmethod
  def version_1(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    shape = x.get_shape().as_list()
    axis = node.attrs.get("axis", 1)
    axis = axis if axis >= 0 else len(shape) + axis

    if axis == len(shape) - 1:
      return [cls.make_tf_tensor(node, **kwargs)]

    cal_shape = (np.prod(shape[0:axis], dtype=np.int64),
                 np.prod(shape[axis:], dtype=np.int64))
    x = tf.reshape(x, cal_shape)

    return [tf.reshape(tf.nn.softmax(x), shape)]
