from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
from onnxtf.backend import run_node
from onnx import helper

class TestStringMethods(unittest.TestCase):
  """ Tests for ops
  """
  def test_relu(self):
    node_def = helper.make_node("Relu", ["X"], ["Y"])
    X = np.random.uniform(-1, 1, 1000)
    output = run_node(node_def, [X])
    np.testing.assert_almost_equal(output["Y"], np.clip(X, 0, 1))

if __name__ == '__main__':
  unittest.main()
