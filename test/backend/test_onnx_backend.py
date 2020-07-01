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

def skip_not_implemented_ops_test(test):
  onnxtf_ops_list = get_onnxtf_supported_ops()
  onnx_ops_list = get_onnx_supported_ops()
  for op in onnx_ops_list:
    op_name = op
    i = 1
    while i < len(op_name):
      if op_name[i].isupper():
        op_name = op_name[:i] + '_' + op_name[i:]
        i += 2
      else:
        i += 1
    if not onnx_ops_list[op]['deprecated']:
      if op in onnxtf_ops_list:
        if onnx_ops_list[op]['version'] not in onnxtf_ops_list[op]:
          test.exclude(r'test_' + op.lower() + '_[a-z,_]*')
          test.exclude(r'test_' + op_name.lower() + '_[a-z,_]*')
      else:
        test.exclude(r'test_' + op.lower() + '_[a-z,_]*')
        test.exclude(r'test_' + op_name.lower() + '_[a-z,_]*')
  return test

# This is a pytest magic variable to load extra plugins
pytest_plugins = 'onnx.backend.test.report',

backend_test = onnx.backend.test.BackendTest(TensorflowBackend, __name__)

# exclude tests of not-implemented-ops
backend_test = skip_not_implemented_ops_test(backend_test)

# manually exclude tests of not-implemented-ops that are using "short name" in their testcase name
# need to remove these lines once those ops support are added into onnx-tf
# temporary exclude StringNormalizer test
backend_test.exclude(r'[a-z,_]*strnorm[a-z,_]*')

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

# TF only support uint8, int32, int64 for indices and int32 for depth in
# tf.one_hot
backend_test.exclude(r'test_onehot_[a-z,_]*')

# TF doesn't support most of the attributes in resize op
# test_node.py will cover the test
backend_test.exclude(r'test_resize_[a-z,_]*')

# range is using loop in the model test but all the outputs datatype are
# missing in the body attribute of the loop
backend_test.exclude(
    r'test_range_float_type_positive_delta_expanded[a-z,_]*')
backend_test.exclude(
    r'test_range_int32_type_negative_delta_expanded[a-z,_]*')

# skip all the cumsum testcases because all the axis in the testcases
# are created as a 1-D 1 element tensor, but the spec clearly state
# that axis should be a 0-D tensor(scalar)
backend_test.exclude(r'test_cumsum_[a-z,_]*')

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

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.enable_report().test_cases)

if __name__ == '__main__':
  unittest.main()
