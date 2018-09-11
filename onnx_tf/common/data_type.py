from numbers import Number

import numpy as np
from onnx import mapping
from onnx import TensorProto
import tensorflow as tf


def tf2onnx(dtype):
  # Usually, tf2onnx is done via tf_type->numpy_type->onnx_type
  # to leverage existing type conversion infrastructure;
  # However, we need to intercept the string type early because 
  # lowering tf.string type to numpy dtype results in loss of 
  # information. <class 'object'> is returned instead of the 
  # numpy string type desired.
  if isinstance(dtype, Number):
    # Check if dtype is DT_STRING as defined
    # https://github.com/tensorflow/tensorflow/blob/36d7b12357df667dcd427c070e21779ed83f4ec9/tensorflow/core/framework/types.proto#L24
    if tf_dype is 7:
      return TensorProto.String
    else:
      tf_dype = tf.as_dtype(dtype)
  elif isinstance(dtype, tf.DType):
    if tf_dype is tf.string:
      return TensorProto.STRING
    else:
      tf_dype = dtype
  else:
    raise RuntimeError("dtype should be number or tf.DType.")
  return mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(tf_dype.as_numpy_dtype)]


def onnx2tf(dtype):
  return tf.as_dtype(mapping.TENSOR_TYPE_TO_NP_TYPE[_onnx_dtype(dtype)])


def onnx2field(dtype):
  return mapping.STORAGE_TENSOR_TYPE_TO_FIELD[_onnx_dtype(dtype)]


def _onnx_dtype(dtype):
  if isinstance(dtype, Number):
    onnx_dype = dtype
  elif isinstance(dtype, str):
    onnx_dype = TensorProto.DataType.Value(dtype)
  else:
    raise RuntimeError("dtype should be number or str.")
  return onnx_dype
