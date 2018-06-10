from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import unittest

import onnx.backend.test

from onnx_tf.backend import TensorflowBackend

# This is a pytest magic variable to load extra plugins
pytest_plugins = 'onnx.backend.test.report',

backend_test = onnx.backend.test.BackendTest(TensorflowBackend, __name__)

# https://github.com/onnx/onnx/issues/349
backend_test.exclude(r'[a-z,_]*GLU[a-z,_]*')

# TF does not support dialation and strides at the same time:
# Will produce strides > 1 not supported in conjunction with dilation_rate > 1
backend_test.exclude(r'[a-z,_]*dilated_strided[a-z,_]*')
backend_test.exclude(r'[a-z,_]*Conv2d_dilated[a-z,_]*')

# Experimental op we do not currently support:
backend_test.exclude(r'[a-z,_]*Upsample[a-z,_]*')

if 'TRAVIS' in os.environ:
  backend_test.exclude('test_vgg19')

major, minor, revision = map(int, onnx.version.version.split("."))
if major == 1 and minor < 2:
	backend_test.exclude('test_operator_add_broadcast_cpu')
	backend_test.exclude('test_operator_add_size1_broadcast_cpu')
	backend_test.exclude('test_operator_add_size1_right_broadcast_cpu')
	backend_test.exclude('test_operator_add_size1_singleton_broadcast_cpu')

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.enable_report().test_cases)

if __name__ == '__main__':
  unittest.main()
