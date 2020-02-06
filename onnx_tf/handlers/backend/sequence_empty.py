import numpy as np
import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.common import data_type
from onnx import mapping

@onnx_op("SequenceEmpty")
#@tf_func(tf.TensorArray)
class SequenceEmpty(BackendHandler):

  @classmethod
  def version_11(cls, node, **kwargs):
    default_dtype = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('float32')]
    dtype = data_type.onnx2tf(node.attrs.get("dtype", default_dtype))
    test = tf.ragged.constant([], dtype = dtype)
    #ta = tf.TensorArray(dtype, size=0, dynamic_size=True, clear_after_read=False, name ='test123') 
    return [test]
