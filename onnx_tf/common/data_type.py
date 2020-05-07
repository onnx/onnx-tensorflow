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
  elif isinstance(dtype, list):
    return [tf2onnx(t) for t in dtype]
  else:
    raise RuntimeError("dtype should be number or tf.DType.")

  # Usually, tf2onnx is done via tf_type->numpy_type->onnx_type
  # to leverage existing type conversion infrastructure;
  # However, we need to intercept the string type early because
  # lowering tf.string type to numpy dtype results in loss of
  # information. <class 'object'> is returned instead of the
  # numpy string type desired.
  if tf_dype is tf.string:
    return TensorProto.STRING

  onnx_dtype = None
  try:
    onnx_dtype = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(
        tf_dype.as_numpy_dtype)]
  finally:
    if onnx_dtype is None:
      common.logger.warning(
          "Can't convert tf dtype {} to ONNX dtype. Return 0 (TensorProto.UNDEFINED)."
          .format(tf_dype))
      onnx_dtype = TensorProto.UNDEFINED
    return onnx_dtype


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


# TODO (tjingrant) unify _onnx_dtype into any_dtype_to_onnx_dtype
def any_dtype_to_onnx_dtype(np_dtype=None, tf_dtype=None, onnx_dtype=None):
  dtype_mask = [1 if val else 0 for val in [np_dtype, tf_dtype, onnx_dtype]]
  num_type_set = sum(dtype_mask)
  assert num_type_set == 1, "One and only one type must be set. However, {} set.".format(
      sum(num_type_set))

  if np_dtype:
    onnx_dtype = mapping.NP_TYPE_TO_TENSOR_TYPE[np_dtype]
  if tf_dtype:
    onnx_dtype = tf2onnx(tf_dtype)

  return onnx_dtype
