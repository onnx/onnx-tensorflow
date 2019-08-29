from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import unittest
import numpy as np
import tensorflow as tf
from onnx_tf.backend import run_node
from onnx_tf.common import supports_device
from onnx_tf.common.legacy import legacy_onnx_pre_ver, legacy_opset_pre_ver
from onnx import helper
from onnx import TensorProto
from onnx import defs


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

  def test_acosh(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support Acosh.".format(
          defs.onnx_opset_version()))
    node_def = helper.make_node("Acosh", ["X"], ["Y"])
    x = self._get_rnd([3, 4, 5])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.arccosh(x))

  def test_add(self):
    node_def = helper.make_node("Add", ["X", "Y"], ["Z"])
    x = self._get_rnd([5, 10, 5, 5])
    y = self._get_rnd([10, 1, 1])
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"],
                                   np.add(x, y.reshape([1, 10, 1, 1])))

    # node_def = helper.make_node("Add", ["A", "B"], ["C"], broadcast=1)
    # a = self._get_rnd([10, 10])
    # b = self._get_rnd([10, 10])
    # output = run_node(node_def, [a, b])
    # np.testing.assert_almost_equal(output["C"], np.add(a, b))

    # node_def = helper.make_node("Add", ["A", "B"], ["C"], broadcast=1)
    # a = self._get_rnd([10, 10])
    # b = self._get_rnd([10,])
    # output = run_node(node_def, [a, b])
    # np.testing.assert_almost_equal(output["C"], np.add(a, b))

  def test_arg_max(self):
    # TODO: need to fix this test
    return
    for axis in [0, 1]:
      node_def = helper.make_node(
          "ArgMax", ["data"], ["reduced"], axis=axis, keepdims=0)
      data = self._get_rnd([10, 10])
      output = run_node(node_def, [data])
      np.testing.assert_almost_equal(output["reduced"],
                                     np.argmax(data, axis=axis))

  def test_arg_min(self):
    # TODO: need to fix this test
    return
    for axis in [0, 1]:
      node_def = helper.make_node(
          "ArgMin", ["data"], ["reduced"], axis=axis, keepdims=0)
      data = self._get_rnd([10, 10])
      output = run_node(node_def, [data])
      np.testing.assert_almost_equal(output["reduced"],
                                     np.argmin(data, axis=axis))

  def test_asinh(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support Asinh.".format(
          defs.onnx_opset_version()))
    node_def = helper.make_node("Asinh", ["X"], ["Y"])
    x = self._get_rnd([3, 4, 5])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.arcsinh(x))

  def test_atanh(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support Atanh.".format(
          defs.onnx_opset_version()))
    node_def = helper.make_node("Atanh", ["X"], ["Y"])
    x = self._get_rnd([3, 4, 5])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.arctanh(x))

  def test_average_pool(self):
    # TODO: fix this test
    return
    device = "CUDA"
    if not supports_device(device):
      raise unittest.SkipTest(
          "Backend doesn't support device {}".format(device))
    shape = [1, 1, 40, 40]
    node_def = helper.make_node(
        "AveragePool", ["X"], ["Y"],
        kernel_shape=[1, 2],
        pads=[1, 1],
        strides=[1, 1])
    x = self._get_rnd(shape)
    output = run_node(node_def, [x], device=device)
    test_output = np.zeros(shape)
    for i1 in range(0, shape[0]):
      for i2 in range(0, shape[1]):
        for j1 in range(0, shape[2]):
          for j2 in range(0, shape[3]):
            test_output[i1][i2][j1][j2] = 0
            count = 0
            for k in range(j2, min(j2 + 2, shape[3])):
              test_output[i1][i2][j1][j2] += x[i1][i2][j1][k]
              count += 1
            test_output[i1][i2][j1][j2] /= count
    np.testing.assert_almost_equal(output["Y"], test_output)

  def _batch_normalization(self, x, mean, variance, bias, scale,
                           variance_epsilon):
    inv = np.reciprocal(np.sqrt(variance + variance_epsilon))
    if scale is not None:
      inv *= scale
    return x * inv + (bias - mean * inv if bias is not None else -mean * inv)

  def test_batch_normalization(self):
    if legacy_opset_pre_ver(6):
      raise unittest.SkipTest("Backend doesn't support consumed flag")
    node_def = helper.make_node(
        "BatchNormalization", ["X", "scale", "bias", "mean", "var"], ["Y"],
        epsilon=0.001)
    x_shape = [3, 5, 4, 2]
    param_shape = [5]
    _param_shape = [1, 5, 1, 1]
    x = self._get_rnd(x_shape, 0, 1)
    m = self._get_rnd(param_shape, 0, 1)
    _m = m.reshape(_param_shape)
    v = self._get_rnd(param_shape, 0, 1)
    _v = v.reshape(_param_shape)
    scale = self._get_rnd(param_shape, 0, 1)
    _scale = scale.reshape(_param_shape)
    bias = self._get_rnd(param_shape, 0, 1)
    _bias = bias.reshape(_param_shape)
    golden = self._batch_normalization(x, _m, _v, _bias, _scale, 0.001)
    output = run_node(node_def, [x, scale, bias, m, v])
    np.testing.assert_almost_equal(output["Y"], golden, decimal=5)

  def test_cast(self):
    if legacy_onnx_pre_ver(1, 2) or legacy_opset_pre_ver(6):
      test_cases = [("FLOAT", tf.float32), ("UINT8", tf.uint8),
                    ("INT8", tf.int8),
                    ("UINT16", tf.uint16), ("INT16", tf.int16),
                    ("INT32", tf.int32), ("INT64", tf.int64), ("BOOL", tf.bool),
                    ("FLOAT16", tf.float16), ("DOUBLE", tf.float64),
                    ("COMPLEX64", tf.complex64), ("COMPLEX128", tf.complex128)]
    else:
      test_cases = [(TensorProto.FLOAT, tf.float32),
                    (TensorProto.UINT8, tf.uint8),
                    (TensorProto.INT8, tf.int8),
                    (TensorProto.UINT16, tf.uint16),
                    (TensorProto.INT16, tf.int16),
                    (TensorProto.INT32, tf.int32),
                    (TensorProto.INT64, tf.int64),
                    (TensorProto.BOOL, tf.bool),
                    (TensorProto.FLOAT16, tf.float16),
                    (TensorProto.DOUBLE, tf.float64),
                    (TensorProto.COMPLEX64, tf.complex64),
                    (TensorProto.COMPLEX128, tf.complex128)]
      if not legacy_opset_pre_ver(9):
         test_cases.append((TensorProto.STRING, tf.string))
    for ty, tf_type in test_cases:
      node_def = helper.make_node("Cast", ["input"], ["output"], to=ty)
      vector = [2, 3]
      output = run_node(node_def, [vector])
      np.testing.assert_equal(output["output"].dtype, tf_type)

    if not legacy_opset_pre_ver(9):
      test_cases2 = [(TensorProto.FLOAT, tf.float32),
                     (TensorProto.INT32, tf.int32),
                     (TensorProto.INT64, tf.int64),
                     (TensorProto.DOUBLE, tf.float64)]
      for ty, tf_type in test_cases2:
        node_def = helper.make_node("Cast", ["input"], ["output"], to=ty)
        vector = ['2', '3']
        output = run_node(node_def, [vector])
        np.testing.assert_equal(output["output"].dtype, tf_type)

  def test_ceil(self):
    node_def = helper.make_node("Ceil", ["X"], ["Y"])
    x = self._get_rnd([1000])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.ceil(x))

  def test_compress(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support Compress.".format(
              defs.onnx_opset_version()))
    axis = 1
    node_def = helper.make_node(
        "Compress", inputs=['X', 'condition'], outputs=['Y'], axis=axis)
    x = self._get_rnd([5, 5, 5])
    cond = np.array([1, 0, 1])
    output = run_node(node_def, inputs=[x, cond])
    np.testing.assert_almost_equal(output['Y'], np.compress(cond, x, axis=axis))

  def test_concat(self):
    shape = [10, 20, 5]
    for axis in range(len(shape)):
      node_def = helper.make_node("Concat", ["X1", "X2"], ["Y"], axis=axis)
      x1 = self._get_rnd(shape)
      x2 = self._get_rnd(shape)
      output = run_node(node_def, [x1, x2])
      np.testing.assert_almost_equal(output["Y"], np.concatenate((x1, x2),
                                                                 axis))

  def test_constant(self):
    shape = [16, 16]
    values = np.random.randn(*shape).flatten().astype(float)
    const2_onnx = helper.make_tensor("const2", TensorProto.DOUBLE, shape,
                                     values)
    node_def = helper.make_node("Constant", [], ["Y"], value=const2_onnx)
    output = run_node(node_def, [])
    np.testing.assert_equal(output["Y"].shape, shape)
    np.testing.assert_almost_equal(output["Y"].flatten(), values)

    # test sparse tensor
    if not legacy_opset_pre_ver(11):
      expected = np.array([[1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]])
      x = np.array([[0, 0], [1, 2]]).flatten().astype(np.int64)
      values = helper.make_tensor("values", TensorProto.INT32, [2], [1, 2])
      indices = helper.make_tensor("indices", TensorProto.INT64, [2, 2], x)
      a = helper.make_sparse_tensor(values, indices,[3, 4])
      node_def = helper.make_node("Constant", [], ["Y"], sparse_value=a)
      output = run_node(node_def, [])
      b = tf.sparse_to_dense(output["Y"].indices, output["Y"].dense_shape, output["Y"].values)
      result = b.eval(session=tf.Session())
      np.testing.assert_equal(result, expected)

  def test_constant_fill(self):
    if not legacy_opset_pre_ver(9):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support ConstantFill.".format(
              defs.onnx_opset_version()))
    shape = [1, 2, 3, 4]
    extra_shape = [5, 6]
    value = 3.
    node_def = helper.make_node(
        "ConstantFill",
        ["X"],
        ["Y"],
        value=value,
        extra_shape=extra_shape,
        dtype=1,
    )
    x = self._get_rnd(shape)
    y = np.zeros(shape + extra_shape)
    y.fill(value)
    output = run_node(node_def, [x])
    np.testing.assert_equal(output["Y"].dtype, tf.float32)
    np.testing.assert_equal(output["Y"], y)

  def test_constant_of_shape(self):
    if defs.onnx_opset_version() < 9:
      raise unittest.SkipTest(
          "ONNX version {} doesn't support ConstantOfShape.".format(
              defs.onnx_opset_version()))
    v = helper.make_tensor("value", TensorProto.FLOAT, [1], [1])
    node_def = helper.make_node("ConstantOfShape", ["X"], ["Y"], value=v)
    x = np.array([4, 3, 2])
    output = run_node(node_def, inputs=[x])
    np.testing.assert_almost_equal(output["Y"], np.ones(x, dtype=np.float32))
    v = helper.make_tensor("value", TensorProto.INT32, [1], [0])
    node_def = helper.make_node("ConstantOfShape", ["X"], ["Y"], value=v)
    x = np.array([10, 6])
    output = run_node(node_def, inputs=[x])
    np.testing.assert_almost_equal(output["Y"], np.zeros(x, dtype=np.int32))

  def test_conv(self):
    device = "CUDA"
    if not supports_device(device):
      raise unittest.SkipTest(
          "Backend doesn't support device {}".format(device))

    N, C, H, W = 4, 3, 5, 5
    x_shape = [N, C, H, W]
    K, kH, kW = 6, 3, 3
    weight_shape = [K, C, kH, kW]
    node_def = helper.make_node(
        "Conv", ["X", "weights"], ["Y"],
        pads=[1, 1, 1, 1],
        kernel_shape=[kH, kW])

    x = self._get_rnd(x_shape)
    weights = self._get_rnd(weight_shape)
    output = run_node(node_def, [x, weights], device=device)

    out_shape = [N, K, H, W]
    test_output = np.zeros(out_shape)
    for n in range(N):
      for c in range(C):
        for h in range(H):
          for w in range(W):
            for k in range(K):
              for kh in range(kH):
                for kw in range(kW):
                  h_in_range = (h - kH // 2 + kh) < H and (
                      h - kH // 2 + kh) >= 0
                  w_in_range = (w - kW // 2 + kw) < W and (
                      w - kW // 2 + kw) >= 0
                  if h_in_range and w_in_range:
                    test_output[n][k][h][w] += (
                        x[n][c][h - kH // 2 + kh][w - kW // 2 + kw] *
                        weights[k][c][kh][kw])

    np.testing.assert_almost_equal(output["Y"], test_output, decimal=5)

  def test_conv_transpose(self):
    # Fix test in the future.
    return
    device = "CUDA"
    if not supports_device(device):
      raise unittest.SkipTest(
          "Backend doesn't support device {}".format(device))
    node_def = helper.make_node(
        "ConvTranspose", ["X", "weights"], ["Y"], pads=[1, 1])
    x_shape = [1, 5, 4]
    x = self._get_rnd(x_shape)
    weight_shape = [5, 3, 2]
    weights = self._get_rnd(weight_shape)
    output = run_node(node_def, [x, weights], device=device)
    out_shape = [x_shape[0], weight_shape[1], x_shape[2]]
    test_output = np.zeros(out_shape)
    for b in range(0, x_shape[0]):
      for m in range(0, weight_shape[1]):
        for h in range(0, x_shape[2]):
          v = 0
          for c in range(0, x_shape[1]):
            for k in range(h, min(h + weight_shape[2], x_shape[2])):
              v += x[b][c][k] * weights[c][m][k - h]
          test_output[b][m][h] = v
    np.testing.assert_almost_equal(output["Y"], test_output, decimal=5)

  def test_cosh(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support Cosh.".format(
          defs.onnx_opset_version()))
    node_def = helper.make_node("Cosh", ["X"], ["Y"])
    x = self._get_rnd([3, 4, 5])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.cosh(x))

  def test_depth_to_space(self):
    node_def = helper.make_node("DepthToSpace", ["X"], ["Y"], blocksize=2)
    x_shape = [1, 12, 1, 1]
    x = self._get_rnd(x_shape)
    output = run_node(node_def, [x])
    x = np.transpose(x, (0, 2, 3, 1))
    y = np.reshape(np.swapaxes(x.reshape(1, 1, 1, 2, 2, 3), 2, 3), (1, 2, 2, 3))
    y = np.transpose(y, (0, 3, 1, 2))
    np.testing.assert_almost_equal(output["Y"], y, decimal=5)

  def test_div(self):
    node_def = helper.make_node("Div", ["X", "Y"], ["Z"])
    x = self._get_rnd([10, 10])
    y = self._get_rnd([10, 10])
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.divide(x, y))

  def test_dropout(self):
    # Since current ONNX only support inference and
    # dropout at inference mode is a no-op,
    # therefore dropout is always a no-op operator
    # in ONNX.
    node_def = helper.make_node("Dropout", ["X"], ["Y"])
    if legacy_opset_pre_ver(7):
      # at inference mode, is_test is always set to 1
      node_def = helper.make_node("Dropout", ["X"], ["Y"], is_test=1)
    x = self._get_rnd([3, 4, 5])
    y = x
    output = run_node(node_def, [x])
    np.testing.assert_equal(output["Y"], y)

  def test_dot(self):
    # this op is removed
    # remove this test in the future
    return
    node_def = helper.make_node("Dot", ["X", "Y"], ["Z"])
    x = np.floor(self._get_rnd([10, 10]))
    y = np.floor(self._get_rnd([10, 10]))
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.dot(x, y))

  def test_elu(self):
    node_def = helper.make_node("Elu", ["X"], ["Y"])
    x = self._get_rnd([100])
    output = run_node(node_def, [x])
    test_output = [self._elu(a) for a in x]
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_equal(self):
    node_def = helper.make_node("Equal", ["X", "Y"], ["Z"])
    x = self._get_rnd([5, 3, 3, 2])
    y = self._get_rnd([3, 3, 1])
    output = run_node(node_def, [x, y])
    np.testing.assert_equal(output["Z"], np.equal(x, np.reshape(
        y, [1, 3, 3, 1])))

  def test_erf(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support Erf.".format(
          defs.onnx_opset_version()))
    node_def = helper.make_node("Erf", ["X"], ["Y"])
    x = self._get_rnd([3, 4, 5])
    output = run_node(node_def, [x])
    exp_output = np.vectorize(math.erf)(x).astype(np.float32)
    np.testing.assert_almost_equal(output["Y"], exp_output)

  def test_exp(self):
    node_def = helper.make_node("Exp", ["X"], ["Y"])
    x = self._get_rnd([100])
    x = x - 3.6
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.exp(x))

  def test_eye_like(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support EyeLike.".format(
          defs.onnx_opset_version()))
    for shape in [[6, 10], [10, 6]]:
      for off_diagonal_offset in [-10, -6, -3, 0, 3, 6, 7, 10]:
        node_def = helper.make_node(
            "EyeLike", ['x'], ['y'], dtype=1, k=off_diagonal_offset)
        x = np.random.randint(0, 100, size=shape, dtype=np.int32)
        y = np.eye(shape[0], shape[1], k=off_diagonal_offset, dtype=np.float32)
        output = run_node(node_def, [x])
        np.testing.assert_equal(output['y'], y)

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
    node_def = helper.make_node(
        "Gemm", ["A", "B", "C"], ["Y"], transA=0, transB=0, alpha=1.0, beta=1.0)
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

  def test_image_sacler(self):
    # Input:  (N x C x H x W), where N is the batch size,
    # C is the number of channels, and H and W are the height
    # and the width of the data
    # Scale: (flout, default 1.0) the scale to apply
    # Bias: applied to each channel, same size as C
    # Output has same shape and type as input
    x = self._get_rnd([1, 3, 224, 224])
    #random distribution over [0,1), so add 0.1
    scale = np.random.rand(1)[0] + 0.1
    bias = np.random.rand(3)
    node_def = helper.make_node(
        "ImageScaler", ["X"], ["Y"], scale=scale, bias=bias)
    output = run_node(node_def, [x])
    test_out = np.multiply(x, scale)
    test_out = np.transpose(test_out, [0, 2, 3, 1])
    test_out = np.add(test_out, bias)
    test_out = np.transpose(test_out, [0, 3, 1, 2])
    np.testing.assert_almost_equal(output["Y"], test_out)

  def test_is_inf(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
      "ONNX version {} doesn't support IsInf.".format(
          defs.onnx_opset_version()))
    input = np.array([-1.2, np.nan, np.inf, 2.8, np.NINF, np.inf],
          dtype=np.float32)
    expected_output = {
      "node_def": np.isinf(input),
      "node_def_neg_false": np.isposinf(input),
      "node_def_pos_false": np.isneginf(input)}
    node_defs = {
      "node_def" :
      helper.make_node("IsInf", ["X"], ["Y"]),
      "node_def_neg_false" :
      helper.make_node("IsInf", ["X"], ["Y"], detect_negative = 0),
      "node_def_pos_false" :
      helper.make_node("IsInf", ["X"], ["Y"], detect_positive = 0)
    }
    for key in node_defs:
      output = run_node(node_defs[key], [input])
      np.testing.assert_equal(output["Y"], expected_output[key])

  def test_isnan(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support IsNaN.".format(
          defs.onnx_opset_version()))
    node_def = helper.make_node("IsNaN", ["X"], ["Y"])
    x = self._get_rnd([3, 3])
    x[0][1] = x[1][0] = x[2][2] = np.nan
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.isnan(x))

  def test_global_lp_pool(self):
    #   Image case:  (N x C x H x W), where N is the batch size,
    # C is the number of channels, and H and W are the height
    # and the width of the data
    #
    #   Non-image case: (N x C x D1 x D2 ... Dn)
    #
    #   Output data tensor from pooling across the input tensor.
    # Dimensions will be N x C x 1 x 1
    node_def = helper.make_node("GlobalLpPool", ["X"], ["Y"])
    x = self._get_rnd([10, 10, 2, 3])
    output = run_node(node_def, [x])
    test_output = np.zeros([10, 10, 1, 1])
    for i1 in range(0, 10):
      for i2 in range(0, 10):
        tmp = np.zeros([2, 3])
        for j1 in range(0, 2):
          for j2 in range(0, 3):
            tmp[j1][j2] = x[i1][i2][j1][j2]
        test_output[i1][i2][0][0] = np.linalg.norm(tmp)
    np.testing.assert_almost_equal(output["Y"], test_output, decimal=5)

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

  def test_less(self):
    node_def = helper.make_node("Less", ["X", "Y"], ["Z"])
    x = self._get_rnd([5, 3, 3, 2])
    y = self._get_rnd([3, 3, 1])
    output = run_node(node_def, [x, y])
    np.testing.assert_equal(output["Z"], np.less(x, np.reshape(y,
                                                               [1, 3, 3, 1])))

  def test_lp_normalization(self):
    node_def = helper.make_node("LpNormalization", ["X"], ["Y"])
    x = self._get_rnd([5, 3, 3, 2])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(
        output["Y"], np.expand_dims(np.linalg.norm(x, axis=-1), -1), rtol=1e-3)

  def test_l_r_n(self):
    # Each input value is divided by:
    #
    # (bias+(alpha/size)*sum(xi^2 for every xi in the local region))^beta
    alpha = 2.0
    beta = 1.0
    bias = 5.0
    size = 3
    node_def = helper.make_node(
        "LRN", ["X"], ["Y"], alpha=alpha, beta=beta, bias=bias, size=size)
    x = self._get_rnd([10, 2, 10, 10])
    output = run_node(node_def, [x])
    test_output = np.zeros([10, 10, 10, 2])
    x = np.transpose(x, axes=[0, 2, 3, 1])
    for i1 in range(0, 10):
      for i2 in range(0, 10):
        for j1 in range(0, 10):
          for j2 in range(0, 2):
            sqr_sum = 0.
            # size of 3 means radius 1 in TF speak
            # i.e. the immediate neighbouring values
            # if "previous" neighbour exists
            if j2 > 0:
              sqr_sum += x[i1][i2][j1][j2 - 1] * x[i1][i2][j1][j2 - 1]
            # current value
            sqr_sum += x[i1][i2][j1][j2] * x[i1][i2][j1][j2]
            # if "next" neighbour exists
            if j2 < 2 - 1:
              sqr_sum += x[i1][i2][j1][j2 + 1] * x[i1][i2][j1][j2 + 1]
            test_output[i1][i2][j1][j2] = \
              x[i1][i2][j1][j2] / ((bias + (alpha * 1. / size) * sqr_sum) ** beta)
    test_output = np.transpose(test_output, axes=[0, 3, 1, 2])
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_floor(self):
    node_def = helper.make_node("Floor", ["X"], ["Y"])
    x = self._get_rnd([100])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.floor(x))

  def test_leakyrelu(self):
    node_def = helper.make_node("LeakyRelu", ["X"], ["Y"], alpha=0.8)
    x = np.floor(self._get_rnd([100]))
    output = run_node(node_def, [x])
    test_output = [self._leaky_relu(a, 0.8) for a in x]
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_log(self):
    node_def = helper.make_node("Log", ["X"], ["Y"])
    x = self._get_rnd([100])
    x = x + 3.6
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
    return
    node_def = helper.make_node(
        "MaxPool", ["X"], ["Y"],
        dilations=[1, 1],
        kernel_shape=[1, 2],
        pads=[0, 0],
        strides=[1, 2])
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

  def test_mean_variance_normalization(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest(
      "ONNX version {} doesn't have test for MeanVarianceNormalization"
      .format(defs.onnx_opset_version()))

    input_data = self._get_rnd([2,2,2,2])
    # Calculate expected output data using formula:
    # (Input - Mean)/SD
    mean = np.mean(input_data, keepdims=1, axis=(0,2,3))
    std = np.std(input_data, keepdims=1, axis=(0,2,3))
    expected_output = (input_data - mean) / std
    # Testing without "axes" argument should default to axes=[0,2,3]
    node_def = helper.make_node("MeanVarianceNormalization", ["X"], ["Y"])
    output = run_node(node_def, [input_data])
    np.testing.assert_almost_equal(output["Y"], expected_output, decimal=5)

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
    node_def = helper.make_node("Mul", ["X", "Y"], ["Z"])
    x = self._get_rnd([5, 10, 5, 5])
    y = self._get_rnd([10, 1, 1])
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"],
                                   np.multiply(x, y.reshape([1, 10, 1, 1])))

  def test_mod(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest("ONNX version {} doesn't support Mod.".format(
          defs.onnx_opset_version()))
    x = self._get_rnd([5, 5])
    y = self._get_rnd([5, 5])
    node_def = helper.make_node("Mod", ["X", "Y"], ["Z"], fmod=0)
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.mod(x, y))
    node_def = helper.make_node("Mod", ["X", "Y"], ["Z"], fmod=1)
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.fmod(x, y))

  def test_neg(self):
    node_def = helper.make_node("Neg", ["X"], ["Y"])
    x = self._get_rnd([1000])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.negative(x))

  def test_non_zero(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support NonZero.".format(
          defs.onnx_opset_version()))
    node_def = helper.make_node("NonZero", ["x"], ["y"])
    x = self._get_rnd([3, 4, 5])
    y = np.array(np.nonzero(x))
    output = run_node(node_def, [x])
    np.testing.assert_equal(output["y"], y)

  def test_onehot(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support OneHot.".format(
          defs.onnx_opset_version()))
    indices = np.array([[0, 2], [1, 2], [0, 1]])
    depth = np.array([5], dtype=np.int32)
    on_value = 6.0
    off_value = 2.0
    values = np.array([off_value, on_value])
    node_def = helper.make_node(
        'OneHot', inputs=['indices', 'depth', 'values'], outputs=['y'], axis=-1)
    y = (np.arange(depth) == indices[..., None]).astype(int)
    y = y * (on_value - off_value) + off_value
    output = run_node(node_def, inputs=[indices, depth, values])
    np.testing.assert_equal(output['y'], y)

  def test_relu(self):
    node_def = helper.make_node("Relu", ["X"], ["Y"])
    x = self._get_rnd([1000])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.maximum(x, 0))

  def test_pad(self):
    node_def = helper.make_node(
        "Pad", ["X"], ["Y"], mode="constant", pads=[1, 1, 1, 1], value=2.0)
    x = self._get_rnd([100, 100])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(
        output["Y"],
        np.lib.pad(x, ((1, 1), (1, 1)), 'constant', constant_values=(2, 2)))

  def test_reciprocal(self):
    node_def = helper.make_node("Reciprocal", ["X"], ["Y"])
    x = self._get_rnd([1000])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], 1.0 / x)

  def test_reduce_l1(self):
    node_def = helper.make_node("ReduceL1", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd([5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"],
                                   np.linalg.norm(x, 1, (1, 2), True))

  def test_reduce_log_sum_exp(self):
    node_def = helper.make_node("ReduceLogSumExp", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd([5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(
        output["Y"],
        np.log(np.sum(np.exp(x), axis=(1, 2), keepdims=True)),
        rtol=1e-3)

  def test_reduce_max(self):
    node_def = helper.make_node("ReduceMax", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd([5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(
        output["Y"], np.max(x, (1, 2), keepdims=True), rtol=1e-3)

  def test_reduce_mean(self):
    node_def = helper.make_node("ReduceMean", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd([5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(
        output["Y"], np.mean(x, (1, 2), keepdims=True), rtol=1e-3)

  def test_reduce_min(self):
    node_def = helper.make_node("ReduceMin", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd([5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(
        output["Y"], np.min(x, (1, 2), keepdims=True), rtol=1e-3)

  def test_reduce_prod(self):
    node_def = helper.make_node("ReduceProd", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd([1, 5, 5, 3])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(
        output["Y"], np.prod(x, (1, 2), keepdims=True), rtol=1e-3)

  def test_reduce_sum(self):
    node_def = helper.make_node("ReduceSum", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd([5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(
        output["Y"], np.sum(x, (1, 2), keepdims=True), rtol=1e-3)

  def test_reduce_sum_square(self):
    node_def = helper.make_node("ReduceSumSquare", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd([5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(
        output["Y"], np.sum(np.square(x), (1, 2), keepdims=True), rtol=1e-3)

  def test_pow(self):
    node_def = helper.make_node("Pow", ["X", "Y"], ["Z"])
    x = self._get_rnd(1000) / 2.0 + 0.5
    y = self._get_rnd(1000) / 2.0 + 0.5
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.power(x, y))

  def test_reshape(self):
    x = self._get_rnd(100)
    shape = [10, 10]
    if defs.onnx_opset_version() < 5:
      node_def = helper.make_node("Reshape", ["X"], ["Z"], shape=shape)
      output = run_node(node_def, [x])
    else:
      node_def = helper.make_node("Reshape", ["X", "Y"], ["Z"])
      output = run_node(node_def, [x, shape])

    np.testing.assert_almost_equal(output["Z"], x.reshape([10, 10]))

  def test_reshape_with_copy(self):
    x = self._get_rnd([10, 20 * 30])
    shape = [0, 20, 30]
    if defs.onnx_opset_version() < 5:
      node_def = helper.make_node("Reshape", ["X"], ["Z"], shape=shape)
      output = run_node(node_def, [x])
    else:
      node_def = helper.make_node("Reshape", ["X", "Y"], ["Z"])
      output = run_node(node_def, [x, shape])

    np.testing.assert_almost_equal(output["Z"], x.reshape([10, 20, 30]))

  def test_selu(self):
    node_def = helper.make_node("Selu", ["X"], ["Y"])
    x = self._get_rnd([1000])
    output = run_node(node_def, [x])
    alpha = 1.6732
    gamma = 1.0507
    x[x <= 0] = gamma * (alpha * np.exp(x[x <= 0]) - alpha)
    x[x > 0] = gamma * x[x > 0]
    np.testing.assert_allclose(output["Y"], x, rtol=1e-3, atol=1e-7)

  def test_shape(self):
    node_def = helper.make_node("Shape", ["X"], ["Y"])
    x = self._get_rnd([5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(output["Y"], np.shape(x))

  def test_shrink(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support Shrink.".format(
              defs.onnx_opset_version()))

    node_def = helper.make_node("Shrink", ["X"], ["Y"], bias=1.5, lambd=1.5)

    X = np.arange(-2.0, 2.1, dtype=np.float32)
    Y = np.array([-0.5, 0, 0, 0, 0.5], dtype=np.float32)
    output = run_node(node_def, [X])
    np.testing.assert_almost_equal(output["Y"], Y)

  def test_sigmoid(self):
    node_def = helper.make_node("Sigmoid", ["X"], ["Y"])
    x = self._get_rnd([1000])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], 1 / (1 + np.exp(-x)))

  def test_sign(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support Sign.".format(
          defs.onnx_opset_version()))
    node_def = helper.make_node("Sign", ["X"], ["Y"])
    x = self._get_rnd([3, 5], -10, 10)
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.sign(x))

  def test_sinh(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support Sinh.".format(
          defs.onnx_opset_version()))
    node_def = helper.make_node("Sinh", ["X"], ["Y"])
    x = self._get_rnd([3, 4, 5])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.sinh(x))

  def test_size(self):
    node_def = helper.make_node("Size", ["X"], ["Y"])
    x = self._get_rnd([5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.size(x))

  def test_slice(self):
    # test case 1 with normal inputs
    axes = [0, 1, 2]
    starts = [0, 0, 0]
    ends = [2, 2, 2]
    steps = [1, 1, 1]

    if legacy_opset_pre_ver(10):
      node_def = helper.make_node(
        "Slice", ["X"], ["S"], axes=axes, starts=starts, ends=ends)
      x = self._get_rnd([1000]).reshape([10, 10, 10])
      output = run_node(node_def, [x])
      np.testing.assert_almost_equal(output["S"], x[0:2, 0:2, 0:2])
    else:
      node_def = helper.make_node(
        "Slice", ["X", "starts", "ends", "axes", "steps"], ["S"])
      x = self._get_rnd([1000]).reshape([10, 10, 10])
      output = run_node(node_def, [x, starts, ends, axes, steps])
      np.testing.assert_almost_equal(output["S"], x[0:2, 0:2, 0:2])

    # test case 2 with negative, out-of-bound and default inputs
    axes = [0, 2]
    starts = [0, -7]
    ends = [-8, 20]

    if legacy_opset_pre_ver(10):
      node_def = helper.make_node(
        "Slice", ["X"], ["S"], axes=axes, starts=starts, ends=ends)
      x = self._get_rnd([1000]).reshape([10, 10, 10])
      output = run_node(node_def, [x])
      np.testing.assert_almost_equal(output["S"], x[0:-8, :, -7:20])
    else:
      node_def = helper.make_node(
        "Slice", ["X", "starts", "ends", "axes"], ["S"])
      x = self._get_rnd([1000]).reshape([10, 10, 10])
      output = run_node(node_def, [x, starts, ends, axes])
      np.testing.assert_almost_equal(output["S"], x[0:-8, :, -7:20])

    # test case 3 with non-default steps
    axes = [0, 1, 2]
    starts = [0, 0, 0]
    ends = [2, 2, 2]
    steps = [2, -2, -1]

    if legacy_opset_pre_ver(10) == False:
      node_def = helper.make_node(
        "Slice", ["X", "starts", "ends", "axes", "steps"], ["S"])
      x = self._get_rnd([1000]).reshape([10, 10, 10])
      output = run_node(node_def, [x, starts, ends, axes, steps])
      np.testing.assert_almost_equal(output["S"], x[0:2:2, 0:2:-2, 0:2:-1])

  def test_softplus(self):
    node_def = helper.make_node("Softplus", ["X"], ["Y"])
    x = self._get_rnd([3, 4, 5])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.log(np.exp(x) + 1))

  def test_softsign(self):
    node_def = helper.make_node("Softsign", ["X"], ["Y"])
    x = self._get_rnd([3, 4, 5])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], x / (1 + np.abs(x)))

  def test_space_to_depth(self):
    node_def = helper.make_node("SpaceToDepth", ["X"], ["Y"], blocksize=2)
    x_shape = [1, 3, 2, 2]
    x = self._get_rnd(x_shape)
    output = run_node(node_def, [x])
    x = np.transpose(x, (0, 2, 3, 1))
    y = np.reshape(
        np.swapaxes(x.reshape(1, 1, 1, 1, 1, 12), 2, 3), (1, 1, 1, 12))
    y = np.transpose(y, (0, 3, 1, 2))
    np.testing.assert_allclose(output["Y"], y, rtol=1e-3)

  def test_split(self):
    split = [3, 3, 4]
    node_def = helper.make_node(
        "Split", ["X"], ["Z%i" % i for i in range(len(split))],
        axis=0,
        split=split)
    x = self._get_rnd([100]).reshape([10, 10])

    output = run_node(node_def, [x])
    for a, b in zip(list(output), np.split(x, np.cumsum(split))[:-1]):
      np.testing.assert_almost_equal(a, b)

  def test_sqrt(self):
    node_def = helper.make_node("Sqrt", ["X"], ["Y"])
    x = self._get_rnd([1000]) + 1.0
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.sqrt(x), decimal=5)

  def test_squeeze(self):
    node_def = helper.make_node("Squeeze", ["X"], ["Y"], axes=[2])
    x = np.array([[[0], [1], [2]]])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.squeeze(x, axis=2))

  def test_sub(self):
    node_def = helper.make_node("Sub", ["X", "Y"], ["Z"])
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

  def test_thresholded_relu(self):
    alpha = 2.0
    node_def = helper.make_node(
        "ThresholdedRelu", ["X"], ["Y"], alpha=alpha)
    x = self._get_rnd([10], -3.0, 3.0)
    y = np.clip(x, alpha, np.inf)
    y[y == alpha] = 0
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], y)

  def test_tile(self):
    if legacy_onnx_pre_ver(1, 2):
      raise unittest.SkipTest(
          "The current version of ONNX does not record correctly the opset of Tile."
      )
    node_def = helper.make_node("Tile", ["X1", "X2"], ["Z"])
    x = self._get_rnd([3, 5, 5, 3])
    repeats = [1, 1, 2, 1]
    output = run_node(node_def, [x, repeats])
    np.testing.assert_allclose(output["Z"], np.tile(x, repeats), rtol=1e-3)

  def test_transpose(self):
    node_def = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 1])
    x = self._get_rnd([1000]).reshape([10, 10, 10])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.transpose(x, (0, 2, 1)))

  def test_topk(self):
    x = np.arange(15, dtype=np.float32).reshape(3, 5)
    values = np.array([[4, 3], [9, 8], [14, 13]], dtype=np.float32)
    indices = np.array([[4, 3],[4, 3],[4, 3]], dtype=np.int64)
    if legacy_opset_pre_ver(10): # for opset = 1
      node_def = helper.make_node("TopK", ["x"], ["values", "indices"], k=2)
      output = run_node(node_def, [x])
    elif legacy_opset_pre_ver(11): # for opset = 10
      k = np.array([2], dtype=np.int64)
      node_def = helper.make_node("TopK", ["x", "k"], ["values", "indices"])
      output = run_node(node_def, [x, k])
    else: # for opset = 11
      x = np.array([[3, 2, 5, 10, 7], [12, 15, 10, 7, 20], [21, 16, 5, 3, 6]],
                   dtype=np.float32)
      values = np.array([[3, 2], [10, 7], [5, 3]], dtype=np.float32)
      indices = np.array([[0, 1], [2, 3], [2, 3]], dtype=np.int64)
      k = np.array([2], dtype=np.int64)
      node_def = helper.make_node(
          "TopK", ["x", "k"], ["values", "indices"], largest=0, sorted=0)
      output = run_node(node_def, [x, k])
    np.testing.assert_almost_equal(output["values"], values)
    np.testing.assert_almost_equal(output["indices"], indices)

  def test_where(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support Where.".format(
              defs.onnx_opset_version()))
    node_def = helper.make_node("Where", ["C","X","Y"], ["Z"])
    c = np.array([[1, 0], [1, 1]], dtype=np.bool)
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[9, 8], [7, 6]], dtype=np.float32)
    output = run_node(node_def, [c,x,y])
    np.testing.assert_almost_equal(output["Z"], np.where(c,x,y))

if __name__ == '__main__':
  unittest.main()
