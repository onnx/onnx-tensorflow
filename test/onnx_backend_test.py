from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import onnx.backend.test
import onnx_tf.backend as tf_backend

# import all test cases at global scope to make them visible to python.unittest
globals().update(onnx.backend.test.BackendTest(tf_backend.TensorflowBackend).test_cases)

if __name__ == '__main__':
    unittest.main()
