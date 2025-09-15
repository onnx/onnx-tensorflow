import numpy as np
import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.common import data_type
from onnx import helper


@onnx_op("SequenceEmpty")
class SequenceEmpty(BackendHandler):

  @classmethod
  def version_11(cls, node, **kwargs):
    default_dtype = helper.np_dtype_to_tensor_dtype(np.dtype('float32'))
    dtype = data_type.onnx2tf(node.attrs.get("dtype", default_dtype))

    ragged = tf.RaggedTensor.from_row_lengths(values=[], row_lengths=[])
    sparse = tf.cast(ragged.to_sparse(), dtype)
    return [tf.RaggedTensor.from_sparse(sparse)]
