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

  def _elu(self, x):
    # f(x) = alpha * (exp(x) - 1.) for x < 0,
    # f(x) = x for x >= 0
    if x < 0.:
      return np.expm1(x)
    return x

  def _leaky_relu(self, x, alpha):
    # f(x) = alpha * x for x < 0,
    # f(x) = x for x >= 0
    if x < 0.:
      return alpha * x
    return x

  def test_div(self):
    node_def = helper.make_node("Div", ["X", "Y"], ["Z"], broadcast=1)
    x = self._get_rnd([10, 10])
    y = self._get_rnd([10, 10])
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.divide(x, y))

  def test_dot(self):
    node_def = helper.make_node("Dot", ["X", "Y"], ["Z"])
    x = np.floor(self._get_rnd([10, 10]));
    y = np.floor(self._get_rnd([10, 10]));
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.dot(x, y))

  def test_elu(self):
    node_def = helper.make_node("Elu", ["X"], ["Y"])
    x = self._get_rnd([100])
    output = run_node(node_def, [x])
    test_output = [self._elu(a) for a in x];
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_exp(self):
    node_def = helper.make_node("Exp", ["X"], ["Y"])
    x = self._get_rnd([100])
    x = x - 3.6;
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.exp(x))

  def test_flatten(self):
    # If input tensor has shape (d_0, d_1, ... d_n) then the
    # output will have shape:
    #
    # (d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn)
    #
    # TODO: pass axis attribute which is supported in newer
    # versions of onnx
    node_def = helper.make_node("Flatten", ["X"], ["Y"])
    x = self._get_rnd([10, 2, 3, 4, 5])
    output = run_node(node_def, [x])
    # TODO: pass axis=3 and uncomment the line below
    # np.testing.assert_almost_equal(output["Y"], x.reshape([60, 20]))
    np.testing.assert_almost_equal(output["Y"], x.reshape([10, 120]))

  def test_gather(self):
    node_def = helper.make_node("Gather", ["X", "Y"], ["Z"])
    x = self._get_rnd([10, 10])
    y = [[0, 1], [1, 2]]
    output = run_node(node_def, [x, y])
    test_output = np.zeros((2, 2, 10))
    for i in range(0, 2):
      for j in range(0, 10):
        test_output[0][i][j] = x[i][j]
    for i in range(0, 2):
      for j in range(0, 10):
        test_output[1][i][j] = x[i + 1][j]
    np.testing.assert_almost_equal(output["Z"], test_output)

  def test_gemm(self):
    # Compute Y = alpha * A * B + beta * C
    node_def = helper.make_node("Gemm", ["A", "B", "C"], ["Y"],
      transA=0, transB=0, broadcast=1, alpha=1.0, beta=1.0)
    x = self._get_rnd([10, 10])
    y = self._get_rnd([10, 10])
    z = self._get_rnd([10, 10])
    output = run_node(node_def, [x, y, z])
    test_output = np.multiply(x, y) + z
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_floor(self):
    node_def = helper.make_node("Floor", ["X"], ["Y"])
    x = self._get_rnd([100])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.floor(x))

  def test_leakyrelu(self):
    node_def = helper.make_node("LeakyRelu", ["X"], ["Y"], alpha=2.0)
    x = np.floor(self._get_rnd([100]))
    output = run_node(node_def, [x])
    test_output = [self._leaky_relu(a, 2.0) for a in x]
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_log(self):
    node_def = helper.make_node("Log", ["X"], ["Y"])
    x = self._get_rnd([100])
    x = x + 3.6;
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.log(x))

  def test_max(self):
    node_def = helper.make_node("Max", ["X1", "X2", "X3", "X4"], ["Z"])
    x1 = self._get_rnd([10, 10])
    x2 = self._get_rnd([10, 10])
    x3 = self._get_rnd([10, 10])
    x4 = self._get_rnd([10, 10])
    output = run_node(node_def, [x1, x2, x3, x4])
    test_output = np.maximum(np.maximum(np.maximum(x1, x2), x3), x4)
    np.testing.assert_almost_equal(output["Z"], test_output)

  def test_min(self):
    node_def = helper.make_node("Min", ["X1", "X2", "X3", "X4"], ["Z"])
    x1 = self._get_rnd([10, 10])
    x2 = self._get_rnd([10, 10])
    x3 = self._get_rnd([10, 10])
    x4 = self._get_rnd([10, 10])
    output = run_node(node_def, [x1, x2, x3, x4])
    test_output = np.minimum(np.minimum(np.minimum(x1, x2), x3), x4)
    np.testing.assert_almost_equal(output["Z"], test_output)

  def test_mul(self):
    node_def = helper.make_node("Mul", ["X", "Y"], ["Z"], broadcast=1)
    x = self._get_rnd([10, 10])
    y = self._get_rnd([10, 10])
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.multiply(x, y))

  def test_neg(self):
    node_def = helper.make_node("Neg", ["X"], ["Y"])
    x = self._get_rnd([1000])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.negative(x))

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
    np.testing.assert_almost_equal(output["Y"], 1.0/x)

  def test_pow(self):
    node_def = helper.make_node("Pow", ["X", "Y"], ["Z"])
    x = self._get_rnd(1000)/2.0 + 0.5
    y = self._get_rnd(1000)/2.0 + 0.5
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"],
                                   np.power(x, y))

  def test_reshape(self):
    node_def = helper.make_node("Reshape", ["X"], ["Y"], shape=[10, 10])
    x = self._get_rnd(100)
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], x.reshape([10, 10]))

  def test_sigmoid(self):
    node_def = helper.make_node("Sigmoid", ["X"], ["Y"])
    x = self._get_rnd([1000])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], 1/(1 + np.exp(-x)))

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
    run_node(helper.make_node("Reshape", ["X"], ["Y"], shape=[5, 25]),
             dummy_inputs_3d[0:1])
    run_node(helper.make_node("Selu", ["X"], ["Y"]), dummy_inputs[0:1])
    run_node(helper.make_node("Sigmoid", ["X"], ["Y"]), dummy_inputs[0:1])
if __name__ == '__main__':
  unittest.main()
