from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re

from onnx import TensorProto
import tensorflow as tf

# refer to
# https://github.com/tensorflow/tensorflow/blob/f284c708a8f3e2672d41dd3e7f6f03b9d26f0c80/tensorflow/core/framework/types.proto
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

TF_TYPE_TO_ONNX_TYPE = {v: k for k, v in
                        ONNX_TYPE_TO_TF_TYPE.items()}

def get_tf_shape_as_list(tf_shape_dim):
  return map(lambda x: x.size, list(tf_shape_dim))

# This function inserts an underscore before every upper
# case letter and lowers that upper case letter except for
# the first letter.
def op_name_to_lower(name):
  return re.sub('(?<!^)(?=[A-Z])', '_', name).lower()