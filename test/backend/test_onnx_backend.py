from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import unittest

import onnx.backend.test

from onnx_tf.backend import TensorflowBackend
from onnx_tf.common.legacy import legacy_onnx_pre_ver
from onnx_tf.common.legacy import legacy_opset_pre_ver


# This is a pytest magic variable to load extra plugins
pytest_plugins = 'onnx.backend.test.report',

backend_test = onnx.backend.test.BackendTest(TensorflowBackend, __name__)

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

if legacy_opset_pre_ver(7):
  backend_test.exclude(r'[a-z,_]*Upsample[a-z,_]*')

if 'TRAVIS' in os.environ:
  backend_test.exclude('test_vgg19')

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

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.enable_report().test_cases)

if __name__ == '__main__':
  unittest.main()
