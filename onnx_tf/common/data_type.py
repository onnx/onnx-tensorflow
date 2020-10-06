from numbers import Number

import numpy as np
from onnx import mapping
from onnx import TensorProto
import tensorflow as tf

import onnx_tf.common as common


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

  if tf_dype is tf.bfloat16:
    return TensorProto.BFLOAT16

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
  # The onnx2tf is done by going to a np type first. However,
  # given that there is no bfloat16 in np at this time, we need
  # to go directly to tf bfloat16 for now.
  if dtype == int(TensorProto.BFLOAT16):
    return tf.as_dtype("bfloat16")
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


def is_safe_cast(from_dtype, to_dtype):
  safe_cast_map = {
      tf.bfloat16: [tf.float32, tf.float64, tf.complex64, tf.complex128],
      tf.float16: [tf.float32, tf.float64, tf.complex64, tf.complex128],
      tf.float32: [tf.float64, tf.complex128],
      tf.float64: [tf.complex128],
      tf.int8: [
          tf.bfloat16, tf.float16, tf.float32, tf.float64, tf.int16, tf.int32,
          tf.int64, tf.complex64, tf.complex128
      ],
      tf.int16: [
          tf.float32, tf.float64, tf.int32, tf.int64, tf.complex64,
          tf.complex128
      ],
      tf.int32: [tf.float64, tf.int64, tf.complex128],
      tf.int64: [],
      tf.uint8: [
          tf.float16, tf.float32, tf.float64, tf.int16, tf.int32, tf.int64,
          tf.complex64, tf.complex128
      ],
      tf.uint16: [
          tf.float32, tf.float64, tf.int32, tf.int64, tf.complex64,
          tf.complex128
      ],
      tf.uint32: [tf.float64, tf.int64, tf.complex128],
      tf.uint64: [],
      tf.complex64: [tf.complex128],
      tf.complex128: []
  }
  return to_dtype in safe_cast_map[from_dtype]


def tf_to_np_str(from_type):
  return mapping.TENSOR_TYPE_TO_NP_TYPE[int(
      tf2onnx(from_type))].name if from_type != tf.bfloat16 else 'bfloat16'


def tf_to_np_str_list(from_list):
  return [tf_to_np_str(from_list[i]) for i in range(len(from_list))]
