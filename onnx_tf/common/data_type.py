from numbers import Number

import numpy as np
from onnx import mapping
from onnx import TensorProto
import tensorflow as tf


def tf2onnx(dtype):
  if isinstance(dtype, Number):
    tf_dype = tf.as_dtype(dtype)
  elif isinstance(dtype, tf.DType):
    tf_dype = dtype
  else:
    raise RuntimeError("dtype should be number or tf.DType.")
  return mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(tf_dype.as_numpy_dtype)]


def onnx2tf(dtype):
  if isinstance(dtype, Number):
    onnx_dype = dtype
  elif isinstance(dtype, str):
    onnx_dype = TensorProto.DataType.Value(dtype)
  else:
    raise RuntimeError("dtype should be number or str.")
  return tf.as_dtype(mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_dype])
