from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
from collections import namedtuple

from onnx import TensorProto
import tensorflow as tf
import numpy as np

# Using the following two functions to prevent shooting ourselves
# in the foot with non-invertible maps.

def invertible(dict):
    # invertible iff one-to-one and onto
    # onto is guaranteed, so check one-to-one
    return len(dict.values()) == len(set(dict.values()))

def invert(dict):
    if not invertible(dict):
        raise ValueError("The dictionary is not invertible"
            " because it is not one-to-one.")
    else:
        inverse = {v: k for k, v in dict.items()}
        return inverse

ONNX_TYPE_TO_TF_TYPE = {
    TensorProto.FLOAT: tf.float32,
    TensorProto.UINT8: tf.uint8,
    TensorProto.INT8: tf.int8,
    TensorProto.UINT16: tf.uint16,
    TensorProto.INT16: tf.int16,
    TensorProto.INT32: tf.int32,
    TensorProto.INT64: tf.int64,
    TensorProto.BOOL: tf.bool,
    TensorProto.FLOAT16: tf.float16,
    TensorProto.DOUBLE: tf.float64,
    TensorProto.COMPLEX64: tf.complex64,
    TensorProto.COMPLEX128: tf.complex128,
    # TODO: uncomment this in the future
    # TensorProto.UINT32: tf.uint32,
    # TensorProto.UINT64: tf.uint64,
}

TF_TYPE_TO_ONNX_TYPE = invert(ONNX_TYPE_TO_TF_TYPE)

ONNX_ATTR_TO_TF_ATTR = {
  "scale": "stddev",
  "high": "maxval",
  "low": "minval",
  "axes": "axis",
  "keepdims": "keep_dims",
  "axis": "dim",
  # TF uses two seeds:
  # seed1: graph level seed
  # seed2: op level seed
  # ONNX only has op level seed, thus the following map
  "seed": "seed2"
  # move this to op specific translator
  # apply only to cast op
  # "to": "dtype",
}

TF_ATTR_TO_ONNX_ATTR = invert(ONNX_ATTR_TO_TF_ATTR)

ONNX_OP_TO_TF_OP = {
  "abs": tf.abs,
  "cast": tf.cast,
  "ceil": tf.ceil,
  "relu": tf.nn.relu,
  "dot": tf.contrib.keras.backend.dot,
  "exp": tf.exp,
  "floor": tf.floor,
  "gather": tf.gather,
  "log": tf.log,
  "neg": tf.negative,
  "pow": tf.pow,
  "random_normal": tf.random_normal,
  "random_uniform": tf.random_uniform,
  "reciprocal": tf.reciprocal,
  "reduce_log_sum_exp": tf.reduce_logsumexp,
  "reduce_max": tf.reduce_max,
  "reduce_mean": tf.reduce_mean,
  "reduce_min": tf.reduce_min,
  "reduce_prod": tf.reduce_prod,
  "reduce_sum": tf.reduce_sum,
  "sigmoid": tf.sigmoid,
  "sqrt": tf.sqrt,
  "squeeze": tf.squeeze,
  "tanh": tf.tanh,
  "transpose": tf.transpose,
  "concat": tf.concat,
}

TF_OP_TO_ONNX_OP = invert(ONNX_OP_TO_TF_OP)

TF_OP_STR_TO_ONNX_OP = {
  "LogicalNot": "Not",
  "Relu": "Relu",
  "Pow": "Pow",
  # TODO:
  # handle Mul, Add, Sub,
  # these are temporarily added to
  # test other ops
  "Mul": "Mul",
  "Add": "Add",

  "Reciprocal": "Reciprocal",
  "Sigmoid": "Sigmoid",
  "Sqrt": "Sqrt",
  "Tanh": "Tanh",
}

def get_tf_shape_as_list(tf_shape_dim):
  return list(map(lambda x: x.size, list(tf_shape_dim)))

# This function inserts an underscore before every upper
# case letter and lowers that upper case letter except for
# the first letter.
def op_name_to_lower(name):
  return re.sub('(?<!^)(?=[A-Z])', '_', name).lower()

# This function is used to convert shape (a list of int)
# from the data format 'NHWC' to 'NCHW'.
# Values of the list will be unchanged, but position of values changes.
def shape_from_nhwc_to_nchw(old_shape, rank):
  if rank >= 3:
    new_shape = old_shape[:]
    for i, s in enumerate(old_shape):
      j = i % (rank-1) + 1 if i != 0 else 0
      new_shape[j] = s
  else:
    new_shape = old_shape
  return new_shape

# This function is used to convert axes (a list of int) 
# from the data format 'NHWC' to 'NCHW'.
# Values of the list will be changed, but position of values will be unchanged.
def axes_from_nhwc_to_nchw(old_axes, rank):
  if rank >= 3:
    new_axes = old_axes[:]
    for i, a in enumerate(old_axes):
      assert a >= -rank and a < rank
      new_axes[i] = a % (rank-1) + 1 if a != 0 else 0
  else:
    new_axes = old_axes
  return new_axes

# This function is used to convert permutation list 
# from the data format 'NHWC' to 'NCHW'.
# Values of the list will be changed, and position of values will be changed too.
def perm_from_nhwc_to_nchw(old_perm, rank):
  if rank >= 3:
    new_perm = old_perm[:]
    for i, p in enumerate(old_perm):
      assert p >= -rank and p < rank
      new_i = i % (rank-1) + 1 if i != 0 else 0
      new_p = p % (rank-1) + 1 if p != 0 else 0
      new_perm[new_i] = new_p
  else:
    new_perm = old_perm
  return new_perm

# This function is used to convert list of padding values for each dim
# from the data format 'NHWC' to 'NCHW'.
# Values of the list will be unchanged, but position of values will be changed.
def pads_from_nhwc_to_nchw(old_pads, rank):
  if rank >= 3:
    new_pads = np.array(old_pads)
    for i, l in enumerate(old_pads):
      for j, p in enumerate(l):
        new_j = j % (rank-1) + 1 if j != 0 else 0
        new_pads[i][new_j] = p
  else:
    new_pads = old_pads
  return new_pads
# This function is used to convert axis (an int) 
# from the data format 'NHWC' to 'NCHW'.
# Value will be changed to match the right axis.
def axis_from_nhwc_to_nchw(old_axis, rank):
  assert old_axis >= -rank and old_axis < rank
  if rank >= 3:
    new_axis = old_axis % (rank-1) + 1 if old_axis != 0 else 0
  else:
    new_axis = old_axis
  return new_axis

# This function is used to convert shape (a list of int)
# from the data format 'NCHW' to 'NHWC'.
# Values of the list will be unchanged, but position of values changes.
def shape_from_nchw_to_nhwc(old_shape, rank):
  if rank >= 3:
    new_shape = np.array(old_shape)
    for i, s in enumerate(old_shape):
      if i == 1:
        new_i = -1
      elif i == 0:
        new_i = 0
      else:
        new_i = i - 1
      new_shape[new_i] = s
  else:
    new_shape = old_shape
  return new_shape

# This function is used to convert axes (a list of int) 
# from the data format 'NCHW' to 'NHWC'.
# Values of the list will be changed, but position of values will be unchanged.
def axes_from_nchw_to_nhwc(old_axes, rank):
  if rank >= 3:
    new_axes = old_axes[:]
    for i, a in enumerate(old_axes):
      if a == 1:
        new_axes[i] = -1
      elif a == 0:
        new_axes[i] = 0
      else:
        new_axes[i] = a - 1
  else:
    new_axes = old_axes
  return new_axes

# This function is used to convert permutation list 
# from the data format 'NCHW' to 'NHWC'.
# Values of the list will be changed, and position of values will be changed too.
def perm_from_nchw_to_nhwc(old_perm, rank):
  if rank >= 3:
    new_perm = old_perm[:]
    for i, p in enumerate(old_perm):
      if p == 1:
        new_i = -1
        new_p = -1
      elif p == 0:
        new_i = 0
        new_p = 0
      else:
        new_i = i - 1
        new_p = p - 1
      new_perm[new_i] = new_p
  else:
    new_perm = old_perm
  return new_perm

# This function is used to convert list of padding values for each dim
# from the data format 'NCHW' to 'NHWC'.
# Values of the list will be unchanged, but position of values will be changed.
def pads_from_nchw_to_nhwc(old_pads, rank):
  if rank >= 3:
    new_pads = np.array(old_pads)
    for i, l in enumerate(old_pads):
      for j, p in enumerate(l):
        if j == 1:
          new_j = -1
        elif j == 0:
          new_j = 0
        else:
          new_j = j - 1
        new_pads[i][new_j] = p
  else:
    new_pads = old_pads
  return new_pads

# This function is used to convert axis (an int) 
# from the data format 'NCHW' to 'NHWC'.
# Value will be changed to match the right axis.
def axis_from_nchw_to_nhwc(old_axis, rank):
  assert old_axis >= -rank and old_axis < rank
  if rank >= 3:
    if old_axis == 1:
      new_axis = -1
    elif old_axis == 0:
      new_axis = 0
    else:
      new_axis = old_axis - 1
  else:
    new_axis = old_axis
  return new_axis
