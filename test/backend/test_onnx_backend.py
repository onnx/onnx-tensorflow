from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import unittest

import onnx.backend.test

from onnx import defs

from onnx_tf import opset_version
from onnx_tf.backend import TensorflowBackend
from onnx_tf.common.legacy import legacy_onnx_pre_ver
from onnx_tf.common.legacy import legacy_opset_pre_ver


def get_onnxtf_supported_ops():
  return opset_version.backend_opset_version


def get_onnx_supported_ops():
  onnx_ops_dict = {}
  for schema in defs.get_all_schemas():
    onnx_ops_dict[schema.name] = {
        'version': schema.since_version,
        'deprecated': schema.deprecated
    }
  return onnx_ops_dict


# This is a pytest magic variable to load extra plugins
pytest_plugins = 'onnx.backend.test.report',

backend_test = onnx.backend.test.runner.Runner(TensorflowBackend, __name__)

# The test cases excluded below should be considered permanent restrictions
# based on the TensorFlow implementation. Unimplemented operators will raise
# a BackendIsNotSupposedToImplementIt exception so that their test cases
# will pass and show a verbose message stating it was effectively skipped.

# https://github.com/onnx/onnx/issues/349
backend_test.exclude(r'[a-z,_]*GLU[a-z,_]*')

# TF does not support dialation and strides at the same time:
# Will produce strides > 1 not supported in conjunction with dilation_rate > 1
backend_test.exclude(r'[a-z,_]*dilated_strided[a-z,_]*')
backend_test.exclude(r'[a-z,_]*Conv2d_dilated[a-z,_]*')

# TF does not have column major max_pool_with_argmax
backend_test.exclude(
    r'[a-z,_]*maxpool_with_argmax_2d_precomputed_strides[a-z,_]*')

# PRelu OnnxBackendPyTorchConvertedModelTest has wrong dim for broadcasting
backend_test.exclude(r'[a-z,_]*PReLU_[0-9]d_multiparam[a-z,_]*')

# TF does not support int8, int16, uint8, uint16, uint32, uint64 for
# tf.floormod and tf.truncatemod
backend_test.exclude(r'test_mod_[a-z,_]*uint[0-9]+')
backend_test.exclude(r'test_mod_[a-z,_]*int(8|(16))+')

# TF doesn't support most of the attributes in resize op
# test_node.py will cover the test
backend_test.exclude(r'test_resize_[a-z,_]*')

# range is using loop in the model test but all the outputs datatype are
# missing in the body attribute of the loop
backend_test.exclude(r'test_range_float_type_positive_delta_expanded[a-z,_]*')
backend_test.exclude(r'test_range_int32_type_negative_delta_expanded[a-z,_]*')

# skip all the cumsum testcases because all the axis in the testcases
# are created as a 1-D 1 element tensor, but the spec clearly state
# that axis should be a 0-D tensor(scalar)
if legacy_opset_pre_ver(13):
  backend_test.exclude(r'test_cumsum_[a-z,_]*')

# Currently ONNX's backend test runner does not support sequence as input/output
backend_test.exclude(r'test_if_seq[a-z,_]*')

# TF session run does not support sequence/RaggedTensor as model inputs
backend_test.exclude(r'test_loop13_seq[a-z,_]*')

# TF minimum/maximum do not support uint64 when auto-cast is False (default)
backend_test.exclude(r'test_min_uint64_[a-z,_]*')
backend_test.exclude(r'test_max_uint64_[a-z,_]*')

if legacy_opset_pre_ver(7):
  backend_test.exclude(r'[a-z,_]*Upsample[a-z,_]*')

if 'TRAVIS' in os.environ:
  backend_test.exclude('test_vgg19')
  backend_test.exclude('zfnet512')

if legacy_onnx_pre_ver(1, 2):
  # These following tests fails by a tiny margin with onnx<1.2:
  backend_test.exclude('test_operator_add_broadcast_cpu')
  backend_test.exclude('test_operator_add_size1_broadcast_cpu')
  backend_test.exclude('test_operator_add_size1_right_broadcast_cpu')
  backend_test.exclude('test_operator_add_size1_singleton_broadcast_cpu')
  backend_test.exclude('test_averagepool_3d_default_cpu')
  # Do not support consumed flag:
  backend_test.exclude('test_batch_normalization')
  # Do not support RNN testing on onnx<1.2 due to incorrect tests:
  backend_test.exclude(r'test_operator_rnn_cpu')
  backend_test.exclude(r'test_operator_lstm_cpu')
  backend_test.exclude(r'test_operator_rnn_single_layer_cpu')

# The onnx test for cast, float to string, does not work
if not legacy_opset_pre_ver(9):
  backend_test.exclude(r'[a-z,_]*cast[a-z,_]*')

if not legacy_opset_pre_ver(10):
  # Do not support dilations != 1 for ConvTranspose, test is added in opset 10
  backend_test.exclude(r'[a-z,_]*convtranspose_dilations[a-z,_]*')

# Concat from sequence with new_axis=1 not supported
backend_test.exclude(r'test_sequence_model5_[a-z,_]*')

# Fails rounding tolerance
backend_test.exclude(r'test_gru_seq_length_[a-z,_]*')

# TF pow does not support uint64 when auto-cast is False (default)
backend_test.exclude(r'test_pow_types_float[0-9]+_uint64+_[a-z,_]*')

# TF session run does not support sequence/RaggedTensor as model inputs
backend_test.exclude(r'test_sequence_insert+_[a-z,_]*')

# Exclude tests for Dropout training that have randomness dependent on
# the different implementations
backend_test.exclude('test_training_dropout_default_[a-z,_]*')
backend_test.exclude('test_training_dropout_[a-z,_]*')
backend_test.exclude('test_training_dropout_default_mask_[a-z,_]*')
backend_test.exclude('test_training_dropout_mask_[a-z,_]*')

# TF module can't run gru, lstm, rnn in one session using custom variables
backend_test.exclude(r'test_gru_[a-z,_]*')
backend_test.exclude(r'test_lstm_[a-z,_]*')
backend_test.exclude(r'test_rnn_[a-z,_]*')
backend_test.exclude(r'test_simple_rnn_[a-z,_]*')

# TF doesn't support auto_pad=SAME_LOWER for Conv and ConvTranspose
backend_test.exclude(r'test_conv_with_autopad_same_[a-z,_]*')
backend_test.exclude(r'test_convtranspose_autopad_same_[a-z,_]*')

# Exclude non-deterministic tests
backend_test.exclude(r'test_bernoulli_expanded[a-z,_]*')
backend_test.exclude(r'test_bernoulli_double_expanded[a-z,_]*')
backend_test.exclude(r'test_bernoulli_seed_expanded[a-z,_]*')

# Exclude optional_get_element, test_optional_has_element tests
backend_test.exclude(r'test_optional_get_element[a-z,_]*')
backend_test.exclude(r'test_optional_has_element[a-z,_]*')

# Exclude BatchNormalization with training_mode=1 tests
backend_test.exclude(r'test_batchnorm_epsilon_training_mode[a-z,_]*')
backend_test.exclude(r'test_batchnorm_example_training_mode[a-z,_]*')

# ONNX 1.9.0 test case does not support sequence
backend_test.exclude(r'[a-z,_]*identity_sequence_[a-z,_]*')

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.enable_report().test_cases)

if __name__ == '__main__':
  unittest.main()
