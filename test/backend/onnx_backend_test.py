from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import unittest

import onnx.backend.test
import onnx_tf.backend as tf_backend

# This is a pytest magic variable to load extra plugins
pytest_plugins = 'onnx.backend.test.report',

backend_test = onnx.backend.test.BackendTest(tf_backend, __name__)

# https://github.com/onnx/onnx/issues/349
backend_test.exclude(r'[a-z,_]*GLU[a-z,_]*')

# https://github.com/onnx/onnx/issues/341
backend_test.exclude(r'[a-z,_]*MaxPool[a-z,_]*')

if 'TRAVIS' in os.environ:
    backend_test.exclude('test_vgg19')

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test
                 .enable_report()
                 .test_cases)

if __name__ == '__main__':
    unittest.main()
