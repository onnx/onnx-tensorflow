from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import tensorflow as tf
from onnx_tf.backend import run_node
from onnx import helper
from onnx.onnx_pb2 import TensorProto

class TestNode(unittest.TestCase):
  """ Tests for nodes
  """

  def _get_rnd(self, shape, low=-1.0, high=1.0):
    return np.random.uniform(low, high, np.prod(shape)) \
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

  def test_abs(self):
    node_def = helper.make_node("Abs", ["X"], ["Y"])
    x = self._get_rnd([1000])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.abs(x))

  def test_add(self):
    node_def = helper.make_node("Add", ["A", "B"], ["C"], broadcast=1)
    a = self._get_rnd([10, 10])
    b = self._get_rnd([10, 10])
    output = run_node(node_def, [a, b])
    np.testing.assert_almost_equal(output["C"], np.add(a, b))

    node_def = helper.make_node("Add", ["A", "B"], ["C"], broadcast=1)
    a = self._get_rnd([10, 10])
    b = self._get_rnd([10,])
    output = run_node(node_def, [a, b])
    np.testing.assert_almost_equal(output["C"], np.add(a, b))

  def test_arg_max(self):
    for axis in [0, 1]:
      node_def = helper.make_node("ArgMax", ["data"], ["reduced"],
                                  axis=axis,
                                  keepdims=0)
      data = self._get_rnd([10, 10])
      output = run_node(node_def, [data])
      np.testing.assert_almost_equal(output["reduced"],
                                     np.argmax(data, axis=axis))

  def test_arg_min(self):
    for axis in [0, 1]:
      node_def = helper.make_node("ArgMin", ["data"], ["reduced"],
                                  axis=axis, keepdims=0)
      data = self._get_rnd([10, 10])
      output = run_node(node_def, [data])
      np.testing.assert_almost_equal(output["reduced"],
                                     np.argmin(data, axis=axis))

  def test_average_pool(self):
    shape = [1, 1, 40, 40]
    node_def = helper.make_node("AveragePool", ["X"], ["Y"],
      kernel_shape=[1,2],
      pads=[0, 0], strides=[1,1])
    x = self._get_rnd(shape)
    output = run_node(node_def, [x], device='CUDA')
    test_output = np.zeros(shape)
    for i1 in range(0, shape[0]):
      for i2 in range(0, shape[1]):
        for j1 in range(0, shape[2]):
          for j2 in range(0, shape[3]):
            test_output[i1][i2][j1][j2] = 0
            count = 0
            for k in range(j2, min(j2+2, shape[3])):
              test_output[i1][i2][j1][j2] += x[i1][i2][j1][k]
              count += 1
            test_output[i1][i2][j1][j2] /= count
    np.testing.assert_almost_equal(output["Y"], test_output)

  def _batch_normalization(self, x, mean, variance, bias, scale,
                           variance_epsilon):
    inv = np.reciprocal(np.sqrt(variance + variance_epsilon))
    if scale is not None:
      inv *= scale
    return x * inv + (bias - mean * inv
                      if bias is not None else -mean * inv)

  def test_batch_normalization(self):
    node_def = helper.make_node("BatchNormalization",
                                ["X", "scale", "bias", "mean", "var"],
                                ["Y"],
                                consumed_inputs=[0, 0, 0, 1, 1],
                                epsilon=0.001)
    x_shape = [3, 5, 4, 2]
    param_shape = [2]
    x = self._get_rnd(x_shape, 0, 1)
    m = self._get_rnd(param_shape, 0, 1)
    v = self._get_rnd(param_shape, 0, 1)
    scale = self._get_rnd(param_shape, 0, 1)
    bias = self._get_rnd(param_shape, 0, 1)
    golden = self._batch_normalization(x, m, v, bias, scale, 0.001)
    output = run_node(node_def, [x, scale, bias, m, v])
    np.testing.assert_almost_equal(output["Y"], golden)

  def test_cast(self):
    for ty, tf_type in [("float", tf.float32),
                        ("uint8", tf.uint8),
                        ("int8", tf.int8),
                        ("uint16", tf.uint16),
                        ("int16", tf.int16),
                        ("int32", tf.int32),
                        ("int64", tf.int64),
                        ("bool", tf.bool),
                        ("float16", tf.float16),
                        ("double", tf.float64),
                        ("complex64", tf.complex64),
                        ("complex128", tf.complex128)]:
      node_def = helper.make_node("Cast", ["input"], ["output"],
                                  to=ty)
      vector = [2, 3]
      output = run_node(node_def, [vector])
      np.testing.assert_equal(output["output"].dtype, tf_type)

  def test_ceil(self):
    node_def = helper.make_node("Ceil", ["X"], ["Y"])
    x = self._get_rnd([1000])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.ceil(x))

  def test_concat(self):
    shape = [10, 20, 5]
    for axis in xrange(len(shape)):
      node_def = helper.make_node("Concat", ["X1", "X2"], ["Y"], axis=axis)
      x1 = self._get_rnd(shape)
      x2 = self._get_rnd(shape)
      output = run_node(node_def, [x1, x2])
      np.testing.assert_almost_equal(output["Y"],
                                     np.concatenate((x1, x2), axis))

  def test_constant(self):
    shape = [10, 20, 9]
    values = np.random.randn(*shape).flatten().astype(float)
    const2_onnx = helper.make_tensor("const2",
                                     TensorProto.FLOAT,
                                     shape,
                                     values)
    node_def = helper.make_node("Constant", [], ["Y"], value=const2_onnx)
    output = run_node(node_def, [])
    np.testing.assert_equal(output["Y"].shape, shape)
    np.testing.assert_almost_equal(output["Y"].flatten(), values)

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
    x = np.floor(self._get_rnd([10, 10]))
    y = np.floor(self._get_rnd([10, 10]))
    z = np.floor(self._get_rnd([10, 10]))
    output = run_node(node_def, [x, y, z])
    test_output = np.matmul(x, y) + z
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_global_average_pool(self):
    #   Image case:  (N x C x H x W), where N is the batch size,
    # C is the number of channels, and H and W are the height
    # and the width of the data
    #
    #   Non-image case: (N x C x D1 x D2 ... Dn)
    #
    #   Output data tensor from pooling across the input tensor.
    # Dimensions will be N x C x 1 x 1
    node_def = helper.make_node("GlobalAveragePool", ["X"], ["Y"])
    x = self._get_rnd([10, 10, 2, 3])
    output = run_node(node_def, [x])
    test_output = np.zeros([10, 10, 1, 1])
    for i1 in range(0, 10):
      for i2 in range(0, 10):
        sum = 0
        for j1 in range(0, 2):
          for j2 in range(0, 3):
            sum += x[i1][i2][j1][j2]
        test_output[i1][i2][0][0] = sum / 6.
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_global_max_pool(self):
    #   Image case:  (N x C x H x W), where N is the batch size,
    # C is the number of channels, and H and W are the height
    # and the width of the data
    #
    #   Non-image case: (N x C x D1 x D2 ... Dn)
    #
    #   Output data tensor from pooling across the input tensor.
    # Dimensions will be N x C x 1 x 1
    node_def = helper.make_node("GlobalMaxPool", ["X"], ["Y"])
    x = self._get_rnd([10, 10, 2, 3])
    output = run_node(node_def, [x])
    test_output = np.zeros([10, 10, 1, 1])
    for i1 in range(0, 10):
      for i2 in range(0, 10):
        max = x[i1][i2][0][0]
        for j1 in range(0, 2):
          for j2 in range(0, 3):
            if max < x[i1][i2][j1][j2]:
              max = x[i1][i2][j1][j2]
        test_output[i1][i2][0][0] = max
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_l_r_n(self):
    # Each input value is divided by:
    #
    # (bias+(alpha/size)*sum(xi^2 for every xi in the local region))^beta
    alpha = 2.0
    beta = 1.0
    bias = 5.0
    size = 3
    node_def = helper.make_node("LRN", ["X"], ["Y"], alpha=alpha,
      beta=beta, bias=bias, size=size)
    x = self._get_rnd([10, 10, 2, 10])
    output = run_node(node_def, [x])
    test_output = np.zeros([10, 10, 2, 10])
    for i1 in range(0, 10):
      for i2 in range(0, 10):
        for j1 in range(0, 2):
          for j2 in range(0, 10):
            sqr_sum = 0.;
            # size of 3 means radius 1 in TF speak
            # i.e. the immediate neighbouring values
            # if "previous" neighbour exists
            if j2 > 0:
              sqr_sum += x[i1][i2][j1][j2 - 1] * x[i1][i2][j1][j2 - 1]
            # current value
            sqr_sum += x[i1][i2][j1][j2] * x[i1][i2][j1][j2]
            # if "next" neighbour exists
            if j2 < 10 - 1:
              sqr_sum += x[i1][i2][j1][j2 + 1] * x[i1][i2][j1][j2 + 1]
            test_output[i1][i2][j1][j2] = \
              x[i1][i2][j1][j2] / ((bias + (alpha * 1. / size) * sqr_sum) ** beta)
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

  def test_max_pool(self):
    node_def = helper.make_node("MaxPool", ["X"], ["Y"],
      dilations=[1,1,1,1], kernel_shape=[1,1,1,2],
      pads=[0,0,0,0], strides=[1,1,1,2])
    x = self._get_rnd([10, 10, 4, 4])
    output = run_node(node_def, [x])
    test_output = np.zeros([10, 10, 4, 2])
    for i1 in range(0, 10):
      for i2 in range(0, 10):
        for j1 in range(0, 4):
          for j2 in range(0, 2):
            test_output[i1][i2][j1][j2] = \
              max(x[i1][i2][j1][2*j2], x[i1][i2][j1][2*j2 + 1])
    np.testing.assert_almost_equal(output["Y"], test_output)

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

  # TODO: better testing for RNN. For now, we are just making sure
  # that it runs.
  def test_optimizedrnn(self):
    node_def = helper.make_node("OptimizedRNN",
                                ["W", "I", "H", "C"],
                                ["O", "OH", "OC"], hidden_size=3, cell_type="lstm")
    x = self._get_rnd([10, 10, 10])
    dummy = np.array([0])
    output = run_node(node_def, [dummy, x, dummy, dummy])

    node_def = helper.make_node("OptimizedRNN",
                                ["W", "I", "H", "C"],
                                ["O", "OH"], hidden_size=3, cell_type="gru")
    x = self._get_rnd([10, 10, 10])
    dummy = np.array([0])
    output = run_node(node_def, [dummy, x, dummy, dummy])

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

  def test_slice(self):
    node_def = helper.make_node("Slice", ["X", "Y", "Z", "W"], ["S"])
    x = self._get_rnd([1000]).reshape([10, 10, 10])
    output = run_node(node_def, [x, [0, 1, 2], [0, 0, 0], [2, 2, 2]])
    np.testing.assert_almost_equal(output["S"], x[0:2, 0:2, 0:2])

  def test_split(self):
    node_def = helper.make_node("Split", ["X", "Y"], ["Z"], axis=0)
    x = self._get_rnd([100]).reshape([10, 10])
    split = [3, 3, 4]
    output = run_node(node_def, [x, split])
    for a, b in zip(output["Z"], np.split(x,np.cumsum(split))[:-1]):
      np.testing.assert_almost_equal(a, b)

  def test_sqrt(self):
    node_def = helper.make_node("Sqrt", ["X"], ["Y"])
    x = self._get_rnd([1000]) + 1.0
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.sqrt(x))

  def test_squeeze(self):
    node_def = helper.make_node("Squeeze", ["X"], ["Y"], axes=[2])
    x = np.array([[[0], [1], [2]]])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"],
                                   np.squeeze(x, axis=2))

  def test_sub(self):
    node_def = helper.make_node("Sub", ["X", "Y"], ["Z"], broadcast=1)
    x = self._get_rnd([10, 10])
    y = self._get_rnd([10, 10])
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.subtract(x, y))

  def test_sum(self):
    node_def = helper.make_node("Sum", ["X1", "X2", "X3", "X4"], ["Z"])
    x1 = self._get_rnd([10, 10])
    x2 = self._get_rnd([10, 10])
    x3 = self._get_rnd([10, 10])
    x4 = self._get_rnd([10, 10])
    output = run_node(node_def, [x1, x2, x3, x4])
    test_output = x1 + x2 + x3 + x4
    np.testing.assert_almost_equal(output["Z"], test_output)

  def test_tanh(self):
    node_def = helper.make_node("Tanh", ["X"], ["Y"])
    x = self._get_rnd([1000]) + 1.0
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.tanh(x), decimal=5)

  def test_transpose(self):
    node_def = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 1])
    x = self._get_rnd([1000]).reshape([10, 10, 10])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.transpose(x, (0, 2, 1)))

if __name__ == '__main__':
  unittest.main()
