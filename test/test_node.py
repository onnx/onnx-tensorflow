from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import tensorflow as tf
import numpy as np
from onnxtf.backend import run_node
from onnx import helper

class TestStringMethods(unittest.TestCase):
  """ Tests for ops
  """

  def _get_rnd(self, shape):
    return np.random.uniform(-1, 1, np.prod(shape)).reshape(shape).astype(np.float32)

  def test_relu(self):
    node_def = helper.make_node("Relu", ["X"], ["Y"])
    X = np.random.uniform(-1, 1, 1000)
    output = run_node(node_def, [X])
    np.testing.assert_almost_equal(output["Y"], np.maximum(X, 0))

  def test_run_all(self):
    dummy_inputs = [self._get_rnd([100]) for i in range(10)]
    run_node(helper.make_node("Relu", ["X"], ["Y"]), dummy_inputs[0:1])
    run_node(helper.make_node("PRelu", ["X", "Slope"], ["Y"]), dummy_inputs[0:2])
    run_node(helper.make_node("Pad", ["X"], ["Y"], mode="constant", paddings=[1,1], value=1.0), dummy_inputs[0:1])

if __name__ == '__main__':
  unittest.main()
