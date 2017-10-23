from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
from onnx_tf.backend import run_node
from onnx import helper
from onnx.onnx_pb2 import TensorProto

class TestStringMethods(unittest.TestCase):
  """ Tests for ops
  """

  def _get_rnd(self, shape):
    return np.random.uniform(-1, 1, np.prod(shape)) \
                      .reshape(shape) \
                      .astype(np.float32)

  def test_relu(self):
    node_def = helper.make_node("Relu", ["X"], ["Y"])
    x = self._get_rnd([1000])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.maximum(x, 0))

  def test_pad(self):
    node_def = helper.make_node("Pad", ["X"], ["Y"],
                                mode="constant",
                                paddings=[1, 1, 1, 1],
                                value=2.0)
    x = self._get_rnd([100, 100])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"],
                                   np.lib.pad(x, ((1, 1), (1, 1)),
                                              'constant',
                                              constant_values=(2, 2)))

  def test_reciprocal(self):
    node_def = helper.make_node("Reciprocal", ["X"], ["Y"])
    x = self._get_rnd([1000])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], 1.0/x);

  def test_pow(self):
    node_def = helper.make_node("Pow", ["X", "Y"], ["Z"])
    x = self._get_rnd(1000)/2.0 + 0.5
    y = self._get_rnd(1000)/2.0 + 0.5
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"],
                                   np.power(x, y))

  def test_run_all(self):
    dummy_inputs = [self._get_rnd([100]) for _ in range(10)]
    dummy_inputs_3d = [self._get_rnd([125]).reshape(5, 5, 5) \
      for _ in range(10)]
    run_node(helper.make_node("Relu", ["X"], ["Y"]), dummy_inputs[0:1])
    run_node(helper.make_node("PRelu", ["X", "Slope"], ["Y"]), \
                                dummy_inputs[0:2])
    run_node(helper.make_node("Pad", ["X"], ["Y"],
                              mode="constant",
                              paddings=[1, 1],
                              value=1.0),
             dummy_inputs[0:1])
    run_node(helper.make_node("Pow", ["X", "Y"], ["Z"]), dummy_inputs[0:2])
    run_node(helper.make_node("RandomNormal",
                              [],
                              ["Y"],
                              dtype=TensorProto.FLOAT,
                              mean=0.0,
                              scale=1.0,
                              shape=[10, 10]),
             [])
    run_node(helper.make_node("RandomNormalLike",
                              ["X"],
                              ["Y"],
                              dtype=TensorProto.FLOAT,
                              mean=0.0,
                              scale=1.0),
             dummy_inputs[0:1])
    run_node(helper.make_node("RandomUniform",
                              [],
                              ["Y"],
                              dtype=TensorProto.FLOAT,
                              low=0.0,
                              high=1.0,
                              shape=[10, 10]),
             [])
    run_node(helper.make_node("RandomUniformLike",
                              ["X"],
                              ["Y"],
                              dtype=TensorProto.FLOAT,
                              low=0.0,
                              high=1.0),
             dummy_inputs[0:1])
    run_node(helper.make_node("Reciprocal", ["X"], ["Y"]), dummy_inputs[0:1])
    for reduce_op in ["LogSumExp", "Max", "Mean", "Min", "Prod", "Sum"]:
      run_node(helper.make_node("Reduce" + reduce_op,
                                ["X"],
                                ["Y"],
                                axes=[0, 1],
                                keepdims=0),
               dummy_inputs_3d[0:1])

if __name__ == '__main__':
  unittest.main()
