from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
from collections import namedtuple

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
}

TF_OP_TO_ONNX_OP = invert(ONNX_OP_TO_TF_OP)

TF_OP_STR_TO_ONNX_OP = {
  "Relu": "Relu",
  "Pow": "Pow",
  # TODO:
  # handle Mul, Add, Sub,
  # these are temporarily added to
  # test other ops
  "Mul": "Mul",
  "Add": "Add",
  "Sub": "Sub",

  "Reciprocal": "Reciprocal",
  "Sigmoid": "Sigmoid",
}

def get_tf_shape_as_list(tf_shape_dim):
  return list(map(lambda x: x.size, list(tf_shape_dim)))

# This function inserts an underscore before every upper
# case letter and lowers that upper case letter except for
# the first letter.
def op_name_to_lower(name):
  return re.sub('(?<!^)(?=[A-Z])', '_', name).lower()
