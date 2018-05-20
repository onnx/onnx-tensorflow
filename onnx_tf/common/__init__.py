from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import uuid

import numpy as np
from onnx import TensorProto
import tensorflow as tf

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
    TensorProto.STRING: tf.string,
    # TODO: uncomment this in the future
    # TensorProto.UINT32: tf.uint32,
    # TensorProto.UINT64: tf.uint64,
}

TF_TYPE_ENUM = [
    "undefined",
    tf.float32,
    tf.uint8,
    tf.int8,
    tf.uint16,
    tf.int16,
    tf.int32,
    tf.int64,
    tf.string,
    tf.bool,
    tf.float16,
    tf.float64,
    tf.complex64,
    tf.complex128,
    # TODO: uncomment this in the future
    # tf.uint32,
    # tf.uint64,
]

ONNX_ATTR_TO_TF_ATTR = {
    "scale": "stddev",
    "high": "maxval",
    "low": "minval",
    "axes": "axis",
    "keepdims": "keep_dims",
}

TF_ATTR_TO_ONNX_ATTR = invert(ONNX_ATTR_TO_TF_ATTR)

ONNX_ATTR_TO_TF_ATTR_PER_OP = {
    "cast": {
        "to": "dtype"
    },
    "depth_to_space": {
        "blocksize": "block_size"
    },
    "gather": {
        "dim": "axis"
    },
    "lp_normalization": {
        "p": "ord"
    },
    "random_normal": {
        "scale": "stddev"
    },
    "random_uniform": {
        "low": "minval",
        "high": "maxval"
    },
}

TF_ATTR_TO_ONNX_ATTR_PER_OP = {
    k: invert(v)
    for k, v in ONNX_ATTR_TO_TF_ATTR_PER_OP.items()
}

ONNX_ATTR_TO_REMOVE_PER_OP = {}

TF_ATTR_TO_REMOVE = [
    "_output_shapes", "T", "seed2", "Tidx", "_class", "Tshape", "Tpaddings",
    "data_format", "transpose_a", "transpose_b", "out_type"
]

ONNX_OP_TO_TF_OP = {
    "abs": tf.abs,
    "cast": tf.cast,
    "ceil": tf.ceil,
    "dot": tf.contrib.keras.backend.dot,
    "exp": tf.exp,
    "floor": tf.floor,
    "gather": tf.gather,
    "identity": tf.identity,
    "log": tf.log,
    "neg": tf.negative,
    "not": tf.logical_not,
    "random_normal": tf.random_normal,
    "random_uniform": tf.random_uniform,
    "reciprocal": tf.reciprocal,
    "reduce_log_sum_exp": tf.reduce_logsumexp,
    "reduce_max": tf.reduce_max,
    "reduce_mean": tf.reduce_mean,
    "reduce_min": tf.reduce_min,
    "reduce_prod": tf.reduce_prod,
    "reduce_sum": tf.reduce_sum,
    "relu": tf.nn.relu,
    "sigmoid": tf.sigmoid,
    "shape": tf.shape,
    "size": tf.size,
    "softplus": tf.nn.softplus,
    "softsign": tf.nn.softsign,
    "sqrt": tf.sqrt,
    "squeeze": tf.squeeze,
    "tanh": tf.tanh,
    "transpose": tf.transpose,
}


def get_tf_shape_as_list(tf_shape_dim):
  return list(map(lambda x: x.size, list(tf_shape_dim)))


# This function inserts an underscore before every upper
# case letter and lowers that upper case letter except for
# the first letter.
def op_name_to_lower(name):
  return re.sub('(?<!^)(?=[A-Z])', '_', name).lower()


def get_attribute_value(attr):
  """ convert Tensorflow AttrValue object to Python object
  """
  if attr.HasField('list'):
    return get_list_value(attr.list)
  if attr.HasField('s'):
    return attr.s
  elif attr.HasField('i'):
    return attr.i
  elif attr.HasField('f'):
    return attr.f
  elif attr.HasField('b'):
    return attr.b
  elif attr.HasField('type'):
    return attr.type
  elif attr.HasField('shape'):
    return attr.type
  elif attr.HasField('tensor'):
    return attr.tensor
  else:
    raise ValueError("Unsupported Tensorflow attribute: {}".format(attr))


def get_list_value(attr):
  """ convert Tensorflow ListValue object to Python object
  """
  if attr.s:
    return attr.s
  elif attr.i:
    return attr.i
  elif attr.f:
    return attr.f
  elif attr.b:
    return attr.b
  elif attr.tensor:
    return attr.tensor
  elif attr.type:
    return attr.type
  elif attr.shape:
    return attr.shape
  elif attr.func:
    return attr.func
  else:
    raise ValueError("Unsupported Tensorflow attribute: {}".format(attr))


def get_unique_suffix():
  return str(uuid.uuid4())[:8]


# Constant string used to indicate that requested padding
# is not natively supported in Tensorflow.
PAD_TF_INCOMPATIBLE = "PAD_TF_INCOMPATIBLE"


def explicit_broadcast(tensor, broadcast_dim):
  total_num_dim = len(tensor.get_shape().as_list())
  if broadcast_dim < 0:
    broadcast_dim += total_num_dim
  dims = [broadcast_dim + i for i in range(len(tensor.shape))]
  for i in range(total_num_dim):
    if i not in dims:
      tensor = tf.expand_dims(tensor, i)


# Until version 6
def get_broadcast_axis(x, y):
  x_shape = x.get_shape().as_list()
  y_shape = y.get_shape().as_list()
  y_dim = len(y_shape)
  if x_shape == y_shape:
    return False
  for i in range(len(x_shape)):
    if x_shape[i:i + y_dim] == y_shape:
      return i