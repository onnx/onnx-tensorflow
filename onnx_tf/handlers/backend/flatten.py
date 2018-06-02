import numpy as np
import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Flatten")
@tf_func(tf.layers.flatten)
class Flatten(BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    axis = node.attrs.get("axis", 1)
    if axis == 1:
      return [cls.make_tensor_from_onnx_node(node, **kwargs)]
    x_shape = kwargs["tensor_dict"][node.inputs[0]].get_shape().as_list()
    shape = (np.prod(x_shape[:axis], dtype=np.int64),
             np.prod(x_shape[axis:], dtype=np.int64))
    return [
        cls.make_tensor_from_onnx_node(
            node, tf_func=tf.reshape, attrs={"shape": shape}, **kwargs)
    ]
