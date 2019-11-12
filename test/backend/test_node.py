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
from onnx_tf.common.pooling_helper import py_pool
from onnx import helper
from onnx import TensorProto
from onnx import defs


class TestNode(unittest.TestCase):
  """ Tests for nodes
  """

  def _get_rnd_float32(self, low=-1.0, high=1.0, shape=None):
    output = np.random.uniform(low, high, shape)
    if shape == None:
      return np.float32(output)
    else:
      return output.astype(np.float32)

  def _get_rnd_int(self, low, high=None, shape=None, dtype=np.int32):
    return np.random.randint(low, high, size=shape, dtype=dtype)

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
    x = self._get_rnd_float32(shape=[1000])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.abs(x))

  def test_acosh(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support Acosh.".format(
          defs.onnx_opset_version()))
    node_def = helper.make_node("Acosh", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[3, 4, 5])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.arccosh(x))

  def test_add(self):
    node_def = helper.make_node("Add", ["X", "Y"], ["Z"])
    x = self._get_rnd_float32(shape=[5, 10, 5, 5])
    y = self._get_rnd_float32(shape=[10, 1, 1])
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
    for axis in [0, 1]:
      node_def = helper.make_node("ArgMax", ["data"], ["reduced"],
                                  axis=axis,
                                  keepdims=0)
      data = self._get_rnd_float32(shape=[10, 10])
      output = run_node(node_def, [data])
      np.testing.assert_almost_equal(output["reduced"],
                                     np.argmax(data, axis=axis))

  def test_arg_min(self):
    for axis in [0, 1]:
      node_def = helper.make_node("ArgMin", ["data"], ["reduced"],
                                  axis=axis,
                                  keepdims=0)
      data = self._get_rnd_float32(shape=[10, 10])
      output = run_node(node_def, [data])
      np.testing.assert_almost_equal(output["reduced"],
                                     np.argmin(data, axis=axis))

  def test_asinh(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support Asinh.".format(
          defs.onnx_opset_version()))
    node_def = helper.make_node("Asinh", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[3, 4, 5])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.arcsinh(x))

  def test_atanh(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support Atanh.".format(
          defs.onnx_opset_version()))
    node_def = helper.make_node("Atanh", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[3, 4, 5])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.arctanh(x))

  def _batch_normalization(self, x, mean, variance, bias, scale,
                           variance_epsilon):
    inv = np.reciprocal(np.sqrt(variance + variance_epsilon))
    if scale is not None:
      inv *= scale
    return x * inv + (bias - mean * inv if bias is not None else -mean * inv)

  def test_batch_normalization(self):
    if legacy_opset_pre_ver(6):
      raise unittest.SkipTest("Backend doesn't support consumed flag")
    node_def = helper.make_node("BatchNormalization",
                                ["X", "scale", "bias", "mean", "var"], ["Y"],
                                epsilon=0.001)
    x_shape = [3, 5, 4, 2]
    param_shape = [5]
    _param_shape = [1, 5, 1, 1]
    x = self._get_rnd_float32(0, 1, shape=x_shape)
    m = self._get_rnd_float32(0, 1, shape=param_shape)
    _m = m.reshape(_param_shape)
    v = self._get_rnd_float32(0, 1, shape=param_shape)
    _v = v.reshape(_param_shape)
    scale = self._get_rnd_float32(0, 1, shape=param_shape)
    _scale = scale.reshape(_param_shape)
    bias = self._get_rnd_float32(0, 1, shape=param_shape)
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
                    (TensorProto.UINT8, tf.uint8), (TensorProto.INT8, tf.int8),
                    (TensorProto.UINT16, tf.uint16),
                    (TensorProto.INT16, tf.int16),
                    (TensorProto.INT32, tf.int32),
                    (TensorProto.INT64, tf.int64), (TensorProto.BOOL, tf.bool),
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
    x = self._get_rnd_float32(shape=[1000])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.ceil(x))

  def test_compress(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support Compress.".format(
              defs.onnx_opset_version()))
    axis = 1
    node_def = helper.make_node("Compress",
                                inputs=['X', 'condition'],
                                outputs=['Y'],
                                axis=axis)
    x = self._get_rnd_float32(shape=[5, 5, 5])
    cond = np.array([1, 0, 1])
    output = run_node(node_def, inputs=[x, cond])
    np.testing.assert_almost_equal(output['Y'], np.compress(cond, x, axis=axis))

  def test_concat(self):
    shape = [10, 20, 5]
    for axis in range(len(shape)):
      node_def = helper.make_node("Concat", ["X1", "X2"], ["Y"], axis=axis)
      x1 = self._get_rnd_float32(shape=shape)
      x2 = self._get_rnd_float32(shape=shape)
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
      a = helper.make_sparse_tensor(values, indices, [3, 4])
      node_def = helper.make_node("Constant", [], ["Y"], sparse_value=a)
      output = run_node(node_def, [])
      b = tf.sparse_to_dense(output["Y"].indices, output["Y"].dense_shape,
                             output["Y"].values)
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
    x = self._get_rnd_float32(shape=shape)
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
    node_def = helper.make_node("Conv", ["X", "weights"], ["Y"],
                                pads=[1, 1, 1, 1],
                                kernel_shape=[kH, kW])

    x = self._get_rnd_float32(shape=x_shape)
    weights = self._get_rnd_float32(shape=weight_shape)
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
                  h_in_range = (h - kH // 2 + kh) < H and (h - kH // 2 +
                                                           kh) >= 0
                  w_in_range = (w - kW // 2 + kw) < W and (w - kW // 2 +
                                                           kw) >= 0
                  if h_in_range and w_in_range:
                    test_output[n][k][h][w] += (
                        x[n][c][h - kH // 2 + kh][w - kW // 2 + kw] *
                        weights[k][c][kh][kw])

    np.testing.assert_almost_equal(output["Y"], test_output, decimal=5)

  def test_conv_integer(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support ConvInteger.".format(
              defs.onnx_opset_version()))

    # Test w_zero_point
    x = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.int8).reshape((1, 1, 3,
                                                                        3))
    w = np.array([2, 2, 2, 2]).astype(np.int8).reshape((1, 1, 2, 2))
    w_zero_point = np.int8(1)
    y = np.array([16, 20, 28, 32]).astype(np.int32).reshape((1, 1, 2, 2))

    node = helper.make_node(
        "ConvInteger", ["X", "W", "w_zero_point"], ["Y"],
        kernel_shape=[2, 2],
        pads=[0, 0, 0, 0],
        dilations=[1, 1])
    output = run_node(node, [x, w, w_zero_point])
    np.testing.assert_almost_equal(output["Y"], y)

    # Test x_zero_point and w_zero_point
    x = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.int8).reshape((1, 1, 3,
                                                                        3))
    x_zero_point = np.int8(1)
    w = np.array([2, 2, 2, 2]).astype(np.int8).reshape((1, 1, 2, 2))
    w_zero_point = np.int8(1)
    y = np.array([12, 16, 24, 28]).astype(np.int32).reshape((1, 1, 2, 2))

    node = helper.make_node(
        "ConvInteger", ["X", "W", "x_zero_point", "w_zero_point"], ["Y"],
        kernel_shape=[2, 2],
        pads=[0, 0, 0, 0],
        dilations=[1, 1])
    output = run_node(node, [x, w, x_zero_point, w_zero_point])
    np.testing.assert_almost_equal(output["Y"], y)

    # Test w_zero_point as 1d tensor
    x = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.int8).reshape((1, 1, 3,
                                                                        3))
    w = np.array([2, 2, 2, 2]).astype(np.int8).reshape((1, 1, 2, 2))
    w_zero_point = np.array([1]).astype(np.int8)
    y = np.array([16, 20, 28, 32]).astype(np.int32).reshape((1, 1, 2, 2))

    node = helper.make_node(
        "ConvInteger", ["X", "W", "w_zero_point"], ["Y"],
        kernel_shape=[2, 2],
        pads=[0, 0, 0, 0],
        dilations=[1, 1])
    output = run_node(node, [x, w, w_zero_point])
    np.testing.assert_almost_equal(output["Y"], y)

    # Test w_zero_point as 1d tensor shape 2
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(np.int8).reshape((1, 1, 3,
                                                                       3))
    w = np.array([2, 2, 2, 2, 2, 2, 2, 2]).astype(np.int8).reshape((2, 1, 2, 2))
    w_zero_point = np.array([1, 2]).astype(np.int8)
    y = np.array([12, 16, 24, 28, 0, 0, 0, 0]).astype(np.int32).reshape((1, 2, 2, 2))

    node = helper.make_node(
        "ConvInteger", ["X", "W", "w_zero_point"], ["Y"],
        kernel_shape=[2, 2],
        pads=[0, 0, 0, 0],
        dilations=[1, 1])
    output = run_node(node, [x, w, w_zero_point])
    np.testing.assert_almost_equal(output["Y"], y)

  def test_conv_transpose(self):
    # Fix test in the future.
    return
    device = "CUDA"
    if not supports_device(device):
      raise unittest.SkipTest(
          "Backend doesn't support device {}".format(device))
    node_def = helper.make_node("ConvTranspose", ["X", "weights"], ["Y"],
                                pads=[1, 1])
    x_shape = [1, 5, 4]
    x = self._get_rnd_float32(shape=x_shape)
    weight_shape = [5, 3, 2]
    weights = self._get_rnd_float32(shape=weight_shape)
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
    x = self._get_rnd_float32(shape=[3, 4, 5])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.cosh(x))

  def test_depth_to_space(self):
    node_def = helper.make_node("DepthToSpace", ["X"], ["Y"], blocksize=2)
    x_shape = [1, 12, 1, 1]
    x = self._get_rnd_float32(shape=x_shape)
    output = run_node(node_def, [x])
    x = np.transpose(x, (0, 2, 3, 1))
    y = np.reshape(np.swapaxes(x.reshape(1, 1, 1, 2, 2, 3), 2, 3), (1, 2, 2, 3))
    y = np.transpose(y, (0, 3, 1, 2))
    np.testing.assert_almost_equal(output["Y"], y, decimal=5)

  def test_dequantize_linear(self):
    node_def = helper.make_node("DequantizeLinear",
                                ["x", "x_scale", "x_zero_point"], ["y"])
    for x, x_zero_point in [[
        self._get_rnd_int(-128, 127, [2, 6], np.int8),
        self._get_rnd_int(-128, 127, dtype=np.int8)
    ],
                            [
                                self._get_rnd_int(0, 255, [2, 6], np.uint8),
                                self._get_rnd_int(0, 255, dtype=np.uint8)
                            ],
                            [self._get_rnd_int(-512, 512, [2, 6]),
                             np.int32(0)]]:
      x_scale = self._get_rnd_float32(-10., 10)
      y = np.subtract(np.float32(x), np.float32(x_zero_point))
      y = np.multiply(y, x_scale)
      output = run_node(node_def, [x, x_scale, x_zero_point])
      np.testing.assert_almost_equal(output["y"], y)

  def test_div(self):
    node_def = helper.make_node("Div", ["X", "Y"], ["Z"])
    x = self._get_rnd_float32(shape=[10, 10])
    y = self._get_rnd_float32(shape=[10, 10])
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
    x = self._get_rnd_float32(shape=[3, 4, 5])
    y = x
    output = run_node(node_def, [x])
    np.testing.assert_equal(output["Y"], y)

  def test_dot(self):
    # this op is removed
    # remove this test in the future
    return
    node_def = helper.make_node("Dot", ["X", "Y"], ["Z"])
    x = np.floor(self._get_rnd_float32(shape=[10, 10]))
    y = np.floor(self._get_rnd_float32(shape=[10, 10]))
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.dot(x, y))

  def test_elu(self):
    node_def = helper.make_node("Elu", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[100])
    output = run_node(node_def, [x])
    test_output = [self._elu(a) for a in x]
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_equal(self):
    node_def = helper.make_node("Equal", ["X", "Y"], ["Z"])
    x = self._get_rnd_float32(shape=[5, 3, 3, 2])
    y = self._get_rnd_float32(shape=[3, 3, 1])
    output = run_node(node_def, [x, y])
    np.testing.assert_equal(output["Z"], np.equal(x,
                                                  np.reshape(y, [1, 3, 3, 1])))

  def test_erf(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support Erf.".format(
          defs.onnx_opset_version()))
    node_def = helper.make_node("Erf", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[3, 4, 5])
    output = run_node(node_def, [x])
    exp_output = np.vectorize(math.erf)(x).astype(np.float32)
    np.testing.assert_almost_equal(output["Y"], exp_output)

  def test_exp(self):
    node_def = helper.make_node("Exp", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[100])
    x = x - 3.6
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.exp(x))

  def test_eye_like(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support EyeLike.".format(
          defs.onnx_opset_version()))
    for shape in [[6, 10], [10, 6]]:
      for off_diagonal_offset in [-10, -6, -3, 0, 3, 6, 7, 10]:
        node_def = helper.make_node("EyeLike", ['x'], ['y'],
                                    dtype=1,
                                    k=off_diagonal_offset)
        x = self._get_rnd_int(0, 100, shape=shape)
        y = np.eye(shape[0], shape[1], k=off_diagonal_offset, dtype=np.float32)
        output = run_node(node_def, [x])
        np.testing.assert_equal(output['y'], y)

  def test_flatten(self):
    shape = [10, 2, 3, 4, 5]
    x = self._get_rnd_float32(shape=shape)
    for axis in range(-len(shape), len(shape)):
      node_def = helper.make_node("Flatten", ["X"], ["Y"], axis=axis)
      output = run_node(node_def, [x])
      if axis == 0:
        new_shape = (1, -1)
      else:
        new_shape = (np.prod(shape[0:axis]).astype(int), -1)
      np.testing.assert_almost_equal(output["Y"], np.reshape(x, new_shape))

  def test_gather(self):
    node_def = helper.make_node("Gather", ["X", "Y"], ["Z"])
    x = self._get_rnd_float32(shape=[10, 10])
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
                                transA=0,
                                transB=0,
                                alpha=1.0,
                                beta=1.0)
    x = np.floor(self._get_rnd_float32(shape=[10, 10]))
    y = np.floor(self._get_rnd_float32(shape=[10, 10]))
    z = np.floor(self._get_rnd_float32(shape=[10, 10]))
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
    x = self._get_rnd_float32(shape=[10, 10, 2, 3])
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

  def test_hardmax(self):
    shape = [2, 3, 4, 5]
    x = self._get_rnd_float32(shape=shape)
    for axis in range(-len(shape), len(shape)):
      node_def = helper.make_node("Hardmax", ["X"], ["Y"], axis=axis)
      output = run_node(node_def, [x])
      shape_in_2d = (np.prod(shape[0:axis]).astype(int),
                     np.prod(shape[axis:len(shape)]))
      x_in_2d = np.reshape(x, shape_in_2d)
      y = np.eye(x_in_2d.shape[1], dtype=x.dtype)[np.argmax(x_in_2d, axis=1)]
      np.testing.assert_almost_equal(output["Y"], np.reshape(y, shape))

  def test_image_sacler(self):
    # Input:  (N x C x H x W), where N is the batch size,
    # C is the number of channels, and H and W are the height
    # and the width of the data
    # Scale: (flout, default 1.0) the scale to apply
    # Bias: applied to each channel, same size as C
    # Output has same shape and type as input
    x = self._get_rnd_float32(shape=[1, 3, 224, 224])
    #random distribution over [0,1), so add 0.1
    scale = np.random.rand(1)[0] + 0.1
    bias = np.random.rand(3)
    node_def = helper.make_node("ImageScaler", ["X"], ["Y"],
                                scale=scale,
                                bias=bias)
    output = run_node(node_def, [x])
    test_out = np.multiply(x, scale)
    test_out = np.transpose(test_out, [0, 2, 3, 1])
    test_out = np.add(test_out, bias)
    test_out = np.transpose(test_out, [0, 3, 1, 2])
    np.testing.assert_almost_equal(output["Y"], test_out)

  def test_is_inf(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest("ONNX version {} doesn't support IsInf.".format(
          defs.onnx_opset_version()))
    input = np.array([-1.2, np.nan, np.inf, 2.8, np.NINF, np.inf],
                     dtype=np.float32)
    expected_output = {
        "node_def": np.isinf(input),
        "node_def_neg_false": np.isposinf(input),
        "node_def_pos_false": np.isneginf(input)
    }
    node_defs = {
        "node_def":
            helper.make_node("IsInf", ["X"], ["Y"]),
        "node_def_neg_false":
            helper.make_node("IsInf", ["X"], ["Y"], detect_negative=0),
        "node_def_pos_false":
            helper.make_node("IsInf", ["X"], ["Y"], detect_positive=0)
    }
    for key in node_defs:
      output = run_node(node_defs[key], [input])
      np.testing.assert_equal(output["Y"], expected_output[key])

  def test_isnan(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support IsNaN.".format(
          defs.onnx_opset_version()))
    node_def = helper.make_node("IsNaN", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[3, 3])
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
    x = self._get_rnd_float32(shape=[10, 10, 2, 3])
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
    x = self._get_rnd_float32(shape=[10, 10, 2, 3])
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
    x = self._get_rnd_float32(shape=[5, 3, 3, 2])
    y = self._get_rnd_float32(shape=[3, 3, 1])
    output = run_node(node_def, [x, y])
    np.testing.assert_equal(output["Z"], np.less(x, np.reshape(y,
                                                               [1, 3, 3, 1])))

  def test_lp_normalization(self):
    for ordr in range(1, 3):
      node_def = helper.make_node("LpNormalization", ["X"], ["Y"], p=ordr)
      x = self._get_rnd_float32(shape=[2, 2, 3, 2])
      output = run_node(node_def, [x])
      np.testing.assert_allclose(
          output["Y"],
          x / np.expand_dims(np.linalg.norm(x, axis=-1, ord=ordr), -1),
          rtol=1e-3)

  def test_l_r_n(self):
    # Each input value is divided by:
    #
    # (bias+(alpha/size)*sum(xi^2 for every xi in the local region))^beta
    alpha = 2.0
    beta = 1.0
    bias = 5.0
    size = 3
    node_def = helper.make_node("LRN", ["X"], ["Y"],
                                alpha=alpha,
                                beta=beta,
                                bias=bias,
                                size=size)
    x = self._get_rnd_float32(shape=[10, 2, 10, 10])
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
    x = self._get_rnd_float32(shape=[100])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.floor(x))

  def test_leakyrelu(self):
    node_def = helper.make_node("LeakyRelu", ["X"], ["Y"], alpha=0.8)
    x = np.floor(self._get_rnd_float32(shape=[100]))
    output = run_node(node_def, [x])
    test_output = [self._leaky_relu(a, 0.8) for a in x]
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_log(self):
    node_def = helper.make_node("Log", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[100])
    x = x + 3.6
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.log(x))

  def test_matmul_integer(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support MatMulInteger.".format(
              defs.onnx_opset_version()))

    node_def = helper.make_node(
        "MatMulInteger", ["A", "B", "a_zero_point", "b_zero_point"], ["Z"])
    lower_bound = {np.uint8: 0, np.int8: -20}
    for dtype in [np.uint8, np.int8]:
      # A & B are 3-D tensor and a_zero_point & b_zero_point are scalar
      A = self._get_rnd_int(
          lower_bound[dtype], 20, shape=(2, 3, 4), dtype=dtype)
      B = self._get_rnd_int(
          lower_bound[dtype], 20, shape=(2, 4, 6), dtype=dtype)
      a_zero_point = self._get_rnd_int(lower_bound[dtype], 20, dtype=dtype)
      b_zero_point = self._get_rnd_int(lower_bound[dtype], 20, dtype=dtype)
      A_minus_zero_point = np.subtract(
          A.astype(np.int32), a_zero_point.astype(np.int32))
      B_minus_zero_point = np.subtract(
          B.astype(np.int32), b_zero_point.astype(np.int32))
      z = np.matmul(A_minus_zero_point, B_minus_zero_point)
      output = run_node(node_def, [A, B, a_zero_point, b_zero_point])
      np.testing.assert_almost_equal(output["Z"], z)
      # A & B are 4-D tensor and a_zero_point & b_zero_point are 1-D tensor
      A = self._get_rnd_int(
          lower_bound[dtype], 20, shape=(2, 5, 3, 4), dtype=dtype)
      B = self._get_rnd_int(
          lower_bound[dtype], 20, shape=(2, 1, 4, 6), dtype=dtype)
      a_zero_point = self._get_rnd_int(
          lower_bound[dtype], 20, shape=(A.shape[-2]), dtype=dtype)
      b_zero_point = self._get_rnd_int(
          lower_bound[dtype], 20, shape=(B.shape[-1]), dtype=dtype)
      a_zero_point_with_reshape = np.reshape(a_zero_point, [A.shape[-2], 1])
      A_minus_zero_point = np.subtract(
          A.astype(np.int32), a_zero_point_with_reshape.astype(np.int32))
      B_minus_zero_point = np.subtract(
          B.astype(np.int32), b_zero_point.astype(np.int32))
      z = np.matmul(A_minus_zero_point, B_minus_zero_point)
      output = run_node(node_def, [A, B, a_zero_point, b_zero_point])
      np.testing.assert_almost_equal(output["Z"], z)

    node_def = helper.make_node("MatMulInteger", ["A", "B"], ["Z"])
    for dtype in [np.uint8, np.int8]:
      # A & B are 3-D tensor
      A = self._get_rnd_int(
          lower_bound[dtype], 20, shape=(2, 3, 4), dtype=dtype)
      B = self._get_rnd_int(
          lower_bound[dtype], 20, shape=(2, 4, 6), dtype=dtype)
      z = np.matmul(A.astype(np.int32), B.astype(np.int32))
      output = run_node(node_def, [A, B])
      np.testing.assert_almost_equal(output["Z"], z)
      # A & B are 4-D tensor
      A = self._get_rnd_int(
          lower_bound[dtype], 20, shape=(2, 5, 3, 4), dtype=dtype)
      B = self._get_rnd_int(
          lower_bound[dtype], 20, shape=(2, 1, 4, 6), dtype=dtype)
      z = np.matmul(A.astype(np.int32), B.astype(np.int32))
      output = run_node(node_def, [A, B])
      np.testing.assert_almost_equal(output["Z"], z)

  def test_max(self):
    node_def = helper.make_node("Max", ["X1", "X2", "X3", "X4"], ["Z"])
    x1 = self._get_rnd_float32(shape=[10, 10])
    x2 = self._get_rnd_float32(shape=[10, 10])
    x3 = self._get_rnd_float32(shape=[10, 10])
    x4 = self._get_rnd_float32(shape=[10, 10])
    output = run_node(node_def, [x1, x2, x3, x4])
    test_output = np.maximum(np.maximum(np.maximum(x1, x2), x3), x4)
    np.testing.assert_almost_equal(output["Z"], test_output)

  def _test_pooling(self, input_shape, kernel_shape, strides=None, 
                    dilations=None, pads=None, auto_pad=None, ceil_mode=None,
                    count_include_pad=None, pooling_type="MAX"):

    op = "MaxPool" if pooling_type.upper().startswith("MAX") else "AveragePool"
    node_def_kwargs = {"op_type": op, "inputs": ["X"], "outputs": ["Y"],
        "kernel_shape": kernel_shape}

    if strides is not None:
        node_def_kwargs["strides"] = strides
    if dilations is not None:
        node_def_kwargs["dilations"] = dilations
    if pads is not None:
        node_def_kwargs["pads"] = pads
    if auto_pad is not None:
        node_def_kwargs["auto_pad"] = auto_pad
        pads = auto_pad
    if ceil_mode is not None:
        node_def_kwargs["ceil_mode"] = ceil_mode
    else:
        ceil_mode = 0
    if count_include_pad is not None:
        node_def_kwargs["count_include_pad"] = count_include_pad

    node_def = helper.make_node(**node_def_kwargs)
 
    x = self._get_rnd_float32(shape=input_shape)
    output = run_node(node_def, [x])

    test_output = py_pool(x, kernel_shape=kernel_shape, strides=strides,
                          dilations=dilations, padding=pads,
                          ceil_mode=ceil_mode, pooling_type=pooling_type,
                          include_indices=False)

    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_max_pool_2d(self):
    kernel_shape=[1, 2]
    strides=[1, 2]

    input_shape = [10, 10, 4, 4]
    self._test_pooling(input_shape=input_shape, kernel_shape=kernel_shape,
                        strides=strides)

  def test_max_pool_2d_same_lower(self):
    kernel_shape=[1, 2]
    strides=[1, 2]
    auto_pad="SAME_LOWER"

    input_shape = [10, 10, 7, 7]
    self._test_pooling(input_shape=input_shape, kernel_shape=kernel_shape,
                        strides=strides, auto_pad=auto_pad)

  def test_max_pool_2d_ceil_same_lower(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support ceil mode.".format(
              defs.onnx_opset_version()))

    kernel_shape=[2, 1]
    strides=[1, 2]
    auto_pad="SAME_LOWER"
    ceil_mode=1

    input_shape = [10, 10, 7, 7]
    self._test_pooling(input_shape=input_shape, kernel_shape=kernel_shape,
                        strides=strides, auto_pad=auto_pad,
                        ceil_mode=ceil_mode)

  def test_max_pool_2d_same_upper(self):
    kernel_shape=[1, 2]
    strides=[1, 2]
    auto_pad="SAME_UPPER"

    input_shape = [10, 10, 7, 7]
    self._test_pooling(input_shape=input_shape, kernel_shape=kernel_shape,
                        strides=strides, auto_pad=auto_pad)

  def test_max_pool_2d_ceil(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support ceil mode.".format(
              defs.onnx_opset_version()))

    kernel_shape = [3, 3]
    strides = [2, 2]
    ceil_mode = 1

    input_shape = [10, 3, 24, 24]
    self._test_pooling(input_shape=input_shape, kernel_shape=kernel_shape,
                        strides=strides, ceil_mode=ceil_mode)

  def test_max_pool_2d_dilations(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support dilations.".format(
              defs.onnx_opset_version()))

    kernel_shape = [3, 3]
    strides = [2, 2]
    dilations = [3, 3]
    node_def = helper.make_node(
        "MaxPool", ["X"], ["Y"],
        kernel_shape=kernel_shape,
        strides=strides,
        dilations=dilations)

    input_shape = [10, 3, 24, 24]
    self._test_pooling(input_shape=input_shape, kernel_shape=kernel_shape,
                        strides=strides, dilations=dilations)


  def test_max_pool_2d_dilations_ceil(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support dilations nor ceil mode.".format(
              defs.onnx_opset_version()))

    kernel_shape = [3, 3]
    strides = [2, 2]
    dilations = [3, 3]
    ceil_mode = 1

    input_shape = [10, 3, 23, 23]
    self._test_pooling(input_shape=input_shape, kernel_shape=kernel_shape,
                        strides=strides, dilations=dilations,
                        ceil_mode=ceil_mode)

  def test_max_pool_2d_dilations_pads(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support dilations.".format(
              defs.onnx_opset_version()))

    kernel_shape = [3, 3]
    strides = [2, 2]
    dilations = [3, 3]
    pads = [1, 1, 2, 2]

    input_shape = [10, 3, 24, 24]
    self._test_pooling(input_shape=input_shape, kernel_shape=kernel_shape,
                       strides=strides, dilations=dilations, pads=pads)

  def test_max_pool_2d_dilations_ceil_pads(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support dilations nor ceil mode.".format(
              defs.onnx_opset_version()))

    kernel_shape = [3, 3]
    strides = [2, 2]
    dilations = [3, 3]
    pads = [1, 1, 2, 2]
    ceil_mode = 1

    input_shape = [10, 3, 23, 23]
    self._test_pooling(input_shape=input_shape, kernel_shape=kernel_shape,
                       strides=strides, dilations=dilations, pads=pads,
                       ceil_mode=ceil_mode)

  def test_max_pool_2d_dilations_same_lower(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support dilations.".format(
              defs.onnx_opset_version()))

    kernel_shape = [3, 3]
    strides = [2, 2]
    dilations = [3, 3]
    auto_pad = "same_lower"

    input_shape = [10, 3, 24, 24]
    self._test_pooling(input_shape=input_shape, kernel_shape=kernel_shape,
                       strides=strides, dilations=dilations,
                       auto_pad=auto_pad)

  def test_max_pool_2d_dilations_same_upper(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support dilations.".format(
              defs.onnx_opset_version()))

    kernel_shape = [2, 3]
    strides = [4, 2]
    dilations = [3, 5]
    auto_pad = "SAME_UPPER"

    input_shape = [10, 3, 24, 24]
    self._test_pooling(input_shape=input_shape, kernel_shape=kernel_shape,
                       strides=strides, dilations=dilations,
                       auto_pad=auto_pad)

  def test_max_pool_3d(self):
    kernel_shape = [3, 3, 3]
    strides = [2, 2, 2]

    input_shape = [10, 3, 23, 23, 23]
    self._test_pooling(input_shape=input_shape, kernel_shape=kernel_shape,
                        strides=strides)

  def test_max_pool_3d_dilations_ceil_pads(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support dilations nor ceil mode.".format(
              defs.onnx_opset_version()))

    kernel_shape = [3, 3, 3]
    strides = [2, 2, 2]
    dilations = [3, 3, 3]
    pads = [1, 1, 2, 2, 1, 1]
    ceil_mode = 1

    input_shape = [10, 3, 23, 23, 23]
    self._test_pooling(input_shape=input_shape, kernel_shape=kernel_shape,
                       strides=strides, dilations=dilations, pads=pads,
                       ceil_mode=ceil_mode)

  def test_max_pool_3d_dilations_same_lower(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support dilations.".format(
              defs.onnx_opset_version()))

    kernel_shape = [3, 1, 2]
    strides = [2, 2, 1]
    dilations = [3, 2, 5]
    auto_pad = "SAME_LOWER"

    input_shape = [10, 3, 23, 23, 23]
    self._test_pooling(input_shape=input_shape, kernel_shape=kernel_shape,
                       strides=strides, dilations=dilations,
                       auto_pad=auto_pad)

  def test_max_pool_1d_dilations_ceil_pads(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support dilations nor ceil mode.".format(
              defs.onnx_opset_version()))

    kernel_shape = [3]
    strides = [2]
    dilations = [3]
    pads = [1, 2]
    ceil_mode = 1

    input_shape = [10, 3, 23]
    self._test_pooling(input_shape=input_shape, kernel_shape=kernel_shape,
                       strides=strides, dilations=dilations, pads=pads,
                       ceil_mode=ceil_mode)

  def test_max_pool_1d(self):
    kernel_shape = [3]
    strides = [2]

    input_shape = [10, 3, 23]
    self._test_pooling(input_shape=input_shape, kernel_shape=kernel_shape,
                       strides=strides)

  def test_max_pool_with_argmax_2d_dilations_ceil_pads(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support dilations nor ceil mode.".format(
              defs.onnx_opset_version()))

    kernel_shape = [3, 3]
    strides = [2, 2]
    dilations = [3, 3]
    pads = [1, 1, 2, 2]
    ceil_mode = True
    node_def = helper.make_node(
        "MaxPool", ["X"], ["Y", "Ind"],
        kernel_shape=kernel_shape,
        strides=strides,
        dilations=dilations,
        pads=pads,
        ceil_mode=ceil_mode)

    input_shape = [10, 1, 23, 23]
    x = self._get_rnd_float32(shape=input_shape)-2
    output = run_node(node_def, [x])

    test_output, test_ind = py_pool(x, kernel_shape=kernel_shape, strides=strides,
                                       dilations=dilations, padding=pads,
                                       ceil_mode=ceil_mode, pooling_type="MAX")

    np.testing.assert_almost_equal(output["Y"], test_output)
    np.testing.assert_almost_equal(output["Ind"], test_ind)

  def test_max_pool_with_argmax_3d(self):
    kernel_shape = [3, 3, 3]
    strides = [2, 2, 2]
    node_def = helper.make_node(
        "MaxPool", ["X"], ["Y", "Ind"],
        kernel_shape=kernel_shape,
        strides=strides)

    input_shape = [10, 1, 23, 23, 23]
    x = self._get_rnd_float32(shape=input_shape)
    self.assertRaises(RuntimeError, run_node, node_def, [x])

  def test_max_pool_4d(self):
    kernel_shape = [3, 3, 3, 3]
    strides = [2, 2, 2, 2]
    node_def = helper.make_node(
        "MaxPool", ["X"], ["Y", "Ind"],
        kernel_shape=kernel_shape,
        strides=strides)

    input_shape = [1, 1, 4, 4, 4, 4]
    x = self._get_rnd_float32(shape=input_shape)
    self.assertRaises(RuntimeError, run_node, node_def, [x])

  def test_average_pool_1d(self):
    kernel_shape = [3]
    strides = [2]

    input_shape = [10, 3, 23]
    self._test_pooling(input_shape=input_shape, kernel_shape=kernel_shape,
                       strides=strides, pooling_type="AVG")
 
  def test_average_pool_2d(self):
    kernel_shape=[1, 2]
    strides=[1, 2]

    input_shape = [10, 10, 4, 4]
    self._test_pooling(input_shape=input_shape, kernel_shape=kernel_shape,
                        strides=strides, pooling_type="AVG")

  def test_average_pool_2d_same_upper_(self):
    kernel_shape=[1, 2]
    strides=[1, 2]
    auto_pad="SAME_UPPER"

    input_shape = [10, 10, 7, 7]
    self._test_pooling(input_shape=input_shape, kernel_shape=kernel_shape,
                        strides=strides, auto_pad=auto_pad, pooling_type="AVG")

  def test_average_pool_3d(self):
    kernel_shape = [3, 3, 3]
    strides = [2, 2, 2]

    input_shape = [10, 3, 23, 23, 23]
    self._test_pooling(input_shape=input_shape, kernel_shape=kernel_shape,
                        strides=strides, pooling_type="AVG")

  def test_mean_variance_normalization(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest(
          "ONNX version {} doesn't have test for MeanVarianceNormalization".
          format(defs.onnx_opset_version()))

    input_data = self._get_rnd_float32(shape=[2, 2, 2, 2])
    # Calculate expected output data using formula:
    # (Input - Mean)/SD
    mean = np.mean(input_data, keepdims=1, axis=(0, 2, 3))
    std = np.std(input_data, keepdims=1, axis=(0, 2, 3))
    expected_output = (input_data - mean) / std
    # Testing without "axes" argument should default to axes=[0,2,3]
    node_def = helper.make_node("MeanVarianceNormalization", ["X"], ["Y"])
    output = run_node(node_def, [input_data])
    np.testing.assert_almost_equal(output["Y"], expected_output, decimal=5)

  def test_min(self):
    node_def = helper.make_node("Min", ["X1", "X2", "X3", "X4"], ["Z"])
    x1 = self._get_rnd_float32(shape=[10, 10])
    x2 = self._get_rnd_float32(shape=[10, 10])
    x3 = self._get_rnd_float32(shape=[10, 10])
    x4 = self._get_rnd_float32(shape=[10, 10])
    output = run_node(node_def, [x1, x2, x3, x4])
    test_output = np.minimum(np.minimum(np.minimum(x1, x2), x3), x4)
    np.testing.assert_almost_equal(output["Z"], test_output)

  def test_mul(self):
    node_def = helper.make_node("Mul", ["X", "Y"], ["Z"])
    x = self._get_rnd_float32(shape=[5, 10, 5, 5])
    y = self._get_rnd_float32(shape=[10, 1, 1])
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"],
                                   np.multiply(x, y.reshape([1, 10, 1, 1])))

  def test_mod(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest("ONNX version {} doesn't support Mod.".format(
          defs.onnx_opset_version()))
    x = self._get_rnd_float32(shape=[5, 5])
    y = self._get_rnd_float32(shape=[5, 5])
    node_def = helper.make_node("Mod", ["X", "Y"], ["Z"], fmod=0)
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.mod(x, y))
    node_def = helper.make_node("Mod", ["X", "Y"], ["Z"], fmod=1)
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.fmod(x, y))

  def test_neg(self):
    node_def = helper.make_node("Neg", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[1000])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.negative(x))

  def test_non_zero(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support NonZero.".format(
          defs.onnx_opset_version()))
    node_def = helper.make_node("NonZero", ["x"], ["y"])
    x = self._get_rnd_float32(shape=[3, 4, 5])
    y = np.array(np.nonzero(x))
    output = run_node(node_def, [x])
    np.testing.assert_equal(output["y"], y)

  def test_onehot(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support OneHot.".format(
          defs.onnx_opset_version()))
    indices = np.array([[0, 2], [1, 2], [0, 1]])
    depth = np.int32(5)
    on_value = 6.0
    off_value = 2.0
    values = np.array([off_value, on_value])
    node_def = helper.make_node('OneHot',
                                inputs=['indices', 'depth', 'values'],
                                outputs=['y'],
                                axis=-1)
    y = (np.arange(depth) == indices[..., None]).astype(int)
    y = y * (on_value - off_value) + off_value
    output = run_node(node_def, inputs=[indices, depth, values])
    np.testing.assert_equal(output['y'], y)

  def test_range(self):
    if legacy_opset_pre_ver(11):
      raise unittest.SkipTest("ONNX version {} doesn't support Range.".format(
          defs.onnx_opset_version()))
    node_def = helper.make_node("Range", ['start', 'limit', 'delta'], ['y'])
    # test positive_delta
    start = self._get_rnd_int(low=0, high=3)
    limit = self._get_rnd_int(low=10, high=30)
    delta = np.int32(3)
    output = run_node(node_def, [start, limit, delta])
    np.testing.assert_equal(output['y'], range(start, limit, delta))
    # test negative_delta
    start = self._get_rnd_int(low=20, high=30)
    limit = self._get_rnd_int(low=1, high=5)
    delta = np.int32(-2)
    output = run_node(node_def, [start, limit, delta])
    np.testing.assert_equal(output['y'], range(start, limit, delta))

  def test_round(self):
    if legacy_opset_pre_ver(11):
      raise unittest.SkipTest("ONNX version {} doesn't support Round.".format(
          defs.onnx_opset_version()))
    node_def = helper.make_node("Round", ["X"], ["Y"])
    x = self._get_rnd_float32(-20.0, 20.0, shape=[1000])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.round(x))

  def test_qLinearMatMul(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support QLinearMatMul.".format(
              defs.onnx_opset_version()))

    def qLinearMatMul(a, a_scale, a_zero_point, b, b_scale, b_zero_point,
                      y_scale, y_zero_point):
      y_dtype = y_zero_point.dtype
      # reshape 1-D a_scale, a_zero_point, y_scale and
      # y_zero_point so it can broadcast in arithmetic
      # operations later
      a_scale_shape = a_scale.shape
      if a_scale_shape and a_scale_shape[0] > 1:
        a_scale = np.reshape(a_scale, [a_scale_shape[0], 1])
        a_zero_point = np.reshape(a_zero_point, [a_scale_shape[0], 1])
      y_scale_shape = y_scale.shape
      if y_scale_shape and y_scale_shape[0] > 1:
        y_scale = np.reshape(y_scale, [y_scale_shape[0], 1])
        y_zero_point = np.reshape(y_zero_point, [y_scale_shape[0], 1])
      # cast everything to float32
      a = a.astype(np.float32)
      a_zero_point = a_zero_point.astype(np.float32)
      b = b.astype(np.float32)
      b_zero_point = b_zero_point.astype(np.float32)
      y_zero_point = y_zero_point.astype(np.float32)
      # dequantize a and b
      dequantized_a = np.subtract(a, a_zero_point)
      dequantized_a = np.multiply(dequantized_a, a_scale)
      dequantized_b = np.subtract(b, b_zero_point)
      dequantized_b = np.multiply(dequantized_b, b_scale)
      # matmul a and b
      x = np.matmul(dequantized_a, dequantized_b)
      # quantize x
      y = np.divide(x, y_scale)
      y = np.round(y)
      y = np.add(y, y_zero_point)
      y = np.clip(y, np.iinfo(y_dtype).min, np.iinfo(y_dtype).max)
      y = y.astype(y_dtype)
      return y

    node_def = helper.make_node('QLinearMatMul', [
        'a', 'a_scale', 'a_zero_point', 'b', 'b_scale', 'b_zero_point',
        'y_scale', 'y_zero_point'
    ], ['y'])
    for dtype in [np.int8, np.uint8]:
      low = np.iinfo(dtype).min
      high = np.iinfo(dtype).max
      a = self._get_rnd_int(low, high, [3, 4, 5, 6], dtype)
      a_scale = self._get_rnd_float32(-0.005, 0.005, [5])
      a_zero_point = self._get_rnd_int(low, high, [5], dtype)
      b = self._get_rnd_int(low, high, [3, 4, 6, 2], dtype)
      b_scale = self._get_rnd_float32(-0.005, 0.005, [2])
      b_zero_point = self._get_rnd_int(low, high, [2], dtype)
      y_scale = self._get_rnd_float32(-0.05, 0.05, [5])
      y_zero_point = self._get_rnd_int(low, high, [5], dtype)
      y = qLinearMatMul(a, a_scale, a_zero_point, b, b_scale, b_zero_point,
                        y_scale, y_zero_point)
      output = run_node(node_def, [
          a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale,
          y_zero_point
      ])
      np.testing.assert_almost_equal(output['y'], y)

  def test_relu(self):
    node_def = helper.make_node("Relu", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[1000])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.maximum(x, 0))

  def test_pad(self):
    x = self._get_rnd_float32(shape=[100, 100])
    if legacy_opset_pre_ver(11):  # for opset = 1 or 2
      # mode = constant
      node_def = helper.make_node(
          "Pad", ["X"], ["Y"], mode="constant", pads=[1, 1, 1, 1], value=2.0)
      output = run_node(node_def, [x])
      y = np.pad(x, ((1, 1), (1, 1)), 'constant', constant_values=(2, 2))
      np.testing.assert_almost_equal(output["Y"], y)
      # mode = reflect and edge
      for mode in ['edge', 'reflect']:
        node_def = helper.make_node(
            "Pad", ["X"], ["Y"], mode=mode, pads=[1, 1, 1, 1])
        output = run_node(node_def, [x])
        y = np.pad(x, ((1, 1), (1, 1)), mode)
        np.testing.assert_almost_equal(output["Y"], y)
    else:  # for opset = 11
      # mode = constant
      node_def = helper.make_node(
          "Pad", ["X", "pads", "constant_values"], ["Y"], mode="constant")
      pads = np.array([1, 1, 1, 1], dtype=np.int64)
      constant_values = 2.0
      output = run_node(node_def, [x, pads, constant_values])
      y = np.pad(x, ((1, 1), (1, 1)), 'constant', constant_values=(2, 2))
      np.testing.assert_almost_equal(output["Y"], y)
      # mode = reflect and edge
      for mode in ['edge', 'reflect']:
        node_def = helper.make_node("Pad", ["X", "pads"], ["Y"], mode=mode)
        output = run_node(node_def, [x, pads])
        y = np.pad(x, ((1, 1), (1, 1)), mode)
        np.testing.assert_almost_equal(output["Y"], y)

  def test_qlinearconv(self):
    if legacy_opset_pre_ver(10):
        raise unittest.SkipTest(
            "ONNX version {} doesn't support QLinearConv.".format(
            defs.onnx_opset_version()))

    # Test w_scale and w_zero_point as scalar
    node_def = helper.make_node("QLinearConv",
        inputs=["x", "x_scale", "x_zero_point", "w", "w_scale", "w_zero_point", "y_scale", "y_zero_point"],
        outputs=["Y"])
    x = np.array([[255, 174, 162, 25, 203, 168, 58],
        [15, 59, 237, 95, 129, 0, 64],
        [56, 242, 153, 221, 168, 12, 166],
        [232, 178, 186, 195, 237, 162, 237],
        [188, 39, 124, 77, 80, 102, 43],
        [127, 230, 21, 83, 41, 40, 134],
        [255, 154, 92, 141, 42, 148, 247], ], dtype=np.uint8).reshape((1, 1, 7, 7))
    x_scale = np.float32(0.00369204697)
    x_zero_point = np.uint8(132)

    w = np.array([0], dtype=np.uint8).reshape((1, 1, 1, 1))
    w_scale = np.float32(0.00172794575)
    w_zero_point = np.uint8(255)

    y = np.array([[0, 81, 93, 230, 52, 87, 197],
        [240, 196, 18, 160, 126, 255, 191],
        [199, 13, 102, 34, 87, 243, 89],
        [23, 77, 69, 60, 18, 93, 18],
        [67, 216, 131, 178, 175, 153, 212],
        [128, 25, 234, 172, 214, 215, 121],
        [0, 101, 163, 114, 213, 107, 8], ], dtype=np.uint8).reshape((1, 1, 7, 7))
    y_scale = np.float32(0.00162681262)
    y_zero_point = np.uint8(123)

    output = run_node(node_def,
            [x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point])
    np.testing.assert_almost_equal(output["Y"], y)

  def test_quantize_linear(self):
    node_def = helper.make_node("QuantizeLinear",
                                ["x", "y_scale", "y_zero_point"], ["y"])
    for x in [
        self._get_rnd_float32(-512., 512., [2, 6]),
        self._get_rnd_int(-512, 512, [2, 6])
    ]:
      y_scale = self._get_rnd_float32(-10., 10.)
      for y_zero_point in [
          self._get_rnd_int(-128, 127, dtype=np.int8),
          self._get_rnd_int(0, 255, dtype=np.uint8)
      ]:
        y = np.divide(x, y_scale)
        y = np.round(y)
        y = np.add(y, y_zero_point)
        if y_zero_point.dtype.type is np.int8:
          y = np.clip(y, -128, 127).astype(np.int8)
        else:
          y = np.clip(y, 0, 255).astype(np.uint8)
        output = run_node(node_def, [x, y_scale, y_zero_point])
        np.testing.assert_almost_equal(output["y"], y)

  def test_reciprocal(self):
    node_def = helper.make_node("Reciprocal", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[1000])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], 1.0 / x)

  def test_reduce_l1(self):
    node_def = helper.make_node("ReduceL1", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd_float32(shape=[5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"],
                                   np.linalg.norm(x, 1, (1, 2), True))

  def test_reduce_log_sum_exp(self):
    node_def = helper.make_node("ReduceLogSumExp", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd_float32(shape=[5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(output["Y"],
                               np.log(
                                   np.sum(np.exp(x), axis=(1, 2),
                                          keepdims=True)),
                               rtol=1e-3)

  def test_reduce_max(self):
    node_def = helper.make_node("ReduceMax", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd_float32(shape=[5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(output["Y"],
                               np.max(x, (1, 2), keepdims=True),
                               rtol=1e-3)

  def test_reduce_mean(self):
    node_def = helper.make_node("ReduceMean", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd_float32(shape=[5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(output["Y"],
                               np.mean(x, (1, 2), keepdims=True),
                               rtol=1e-3)

  def test_reduce_min(self):
    node_def = helper.make_node("ReduceMin", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd_float32(shape=[5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(output["Y"],
                               np.min(x, (1, 2), keepdims=True),
                               rtol=1e-3)

  def test_reduce_prod(self):
    node_def = helper.make_node("ReduceProd", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd_float32(shape=[1, 5, 5, 3])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(output["Y"],
                               np.prod(x, (1, 2), keepdims=True),
                               rtol=1e-3)

  def test_reduce_sum(self):
    node_def = helper.make_node("ReduceSum", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd_float32(shape=[5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(output["Y"],
                               np.sum(x, (1, 2), keepdims=True),
                               rtol=1e-3)

  def test_reduce_sum_square(self):
    node_def = helper.make_node("ReduceSumSquare", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd_float32(shape=[5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(output["Y"],
                               np.sum(np.square(x), (1, 2), keepdims=True),
                               rtol=1e-3)

  def test_pow(self):
    node_def = helper.make_node("Pow", ["X", "Y"], ["Z"])
    x = self._get_rnd_float32(shape=1000) / 2.0 + 0.5
    y = self._get_rnd_float32(shape=1000) / 2.0 + 0.5
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.power(x, y))

  def test_reshape(self):
    x = self._get_rnd_float32(shape=100)
    shape = [10, 10]
    if defs.onnx_opset_version() < 5:
      node_def = helper.make_node("Reshape", ["X"], ["Z"], shape=shape)
      output = run_node(node_def, [x])
    else:
      node_def = helper.make_node("Reshape", ["X", "Y"], ["Z"])
      output = run_node(node_def, [x, shape])

    np.testing.assert_almost_equal(output["Z"], x.reshape([10, 10]))

  def test_reshape_with_copy(self):
    x = self._get_rnd_float32(shape=[10, 20 * 30])
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
    x = self._get_rnd_float32(shape=[1000])
    output = run_node(node_def, [x])
    alpha = 1.6732
    gamma = 1.0507
    x[x <= 0] = gamma * (alpha * np.exp(x[x <= 0]) - alpha)
    x[x > 0] = gamma * x[x > 0]
    np.testing.assert_allclose(output["Y"], x, rtol=1e-3, atol=1e-7)

  def _run_scan_node(self, initial, x1, x2, input_shape, output_shape,
                     scan_input_axes=None, scan_input_directions=None,
                     scan_output_axes=None, scan_output_directions=None,
                     sequence_lens=None, directions=None):
    """
      Subgraph looks like this.

      [const1]       state_in                    concat1_in    concat2_in_
            \           |                                 \     /
             \--------[Add]                               [Concat]
                        |                                    |
                        |                                 concat_out
                        |                                    |
                        |                                  [Add]----------[const1]
                        |                                    |
                        |                                add_out_1
                        |                                    |
                        |                                 [Split]
                        |                               /  |   |   \
                   state_out                  split1_out    ...     split4_out
    """ 
    val_1 = helper.make_tensor(
        name='const_tensor',
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=[1],
    )
    constant_node = helper.make_node("Constant", [], ["const_1"], value=val_1)
    state_add_node = helper.make_node("Add", ["state_in", "const_1"],
                                      ["state_out"])
    concat_node = helper.make_node("Concat", ["concat1_in", "concat2_in"],
                                   ["concat_out"], axis=0)
    add_node = helper.make_node("Add", ["concat_out", "const_1"], ["add_out"])
    split_node = helper.make_node("Split", ["add_out"],
                                  ["split1_out", "split2_out", "split3_out",
                                   "split4_out"])

    state_in = helper.make_tensor_value_info('state_in',
                                             TensorProto.FLOAT, [1])
    concat1_in = helper.make_tensor_value_info('concat1_in',
                                               TensorProto.FLOAT, input_shape)
    concat2_in = helper.make_tensor_value_info('concat2_in',
                                               TensorProto.FLOAT, input_shape)
    state_out  = helper.make_tensor_value_info('state_out',
                                               TensorProto.FLOAT, [1])
    split1_out = helper.make_tensor_value_info('split1_out',
                                               TensorProto.FLOAT, output_shape)
    split2_out = helper.make_tensor_value_info('split2_out',
                                               TensorProto.FLOAT, output_shape)
    split3_out = helper.make_tensor_value_info('split3_out',
                                               TensorProto.FLOAT, output_shape)
    split4_out = helper.make_tensor_value_info('split4_out',
                                               TensorProto.FLOAT, output_shape)

    scan_body = helper.make_graph(
        [constant_node, state_add_node, concat_node, add_node, split_node],
        "scan_body",
        [state_in, concat1_in, concat2_in],
        [state_out, split1_out, split2_out, split3_out, split4_out],
    )

    node_kwargs = {"op_type": "Scan", 
                   "inputs": ["initial", "x1", "x2"],
                   "outputs": ["y", "z1", "z2", "z3", "z4"],
                   "num_scan_inputs": 2,
                   "body": scan_body
                  }
    if sequence_lens is not None:
        node_kwargs["inputs"] = ["" if sequence_lens is str else 
                                 "seq_lens"] + node_kwargs["inputs"]

    if scan_input_axes is not None:
      node_kwargs["scan_input_axes"] = scan_input_axes
    if scan_input_directions is not None:
      node_kwargs["scan_input_directions"] = scan_input_directions
    if scan_output_axes is not None:
      node_kwargs["scan_output_axes"] = scan_output_axes
    if scan_output_directions is not None:
      node_kwargs["scan_output_directions"] = scan_output_directions
    if directions is not None:
      node_kwargs["directions"] = directions

    scan_node = helper.make_node(**node_kwargs)

    if sequence_lens is None:
        inputs = [initial, x1, x2]
    else:
        inputs = [sequence_lens, initial, x1, x2]

    return run_node(scan_node, inputs)

  def test_scan_v8(self):
    if legacy_opset_pre_ver(8) or not legacy_opset_pre_ver(9):
      raise unittest.SkipTest(
          "ONNX version {} not supported.".format(
              defs.onnx_opset_version()))

    initial = self._get_rnd_int(0, 100, shape=[5, 1]).astype(np.float32)
    x1 = self._get_rnd_float32(0, 1000, shape=[5, 20, 6, 2])
    x2 = self._get_rnd_float32(0, 1000, shape=[5, 20, 6, 2])

    directions = [0, 1]
    sequence_lens = np.array([15, 20, 14, 18, 20]).astype(np.int32)

    Y = initial + (np.shape(x1)[1] if sequence_lens is str else \
                  np.reshape(sequence_lens,[-1, 1]))
    x1_out = x1 + 1
    # left-right flip x2 (reverse direction)
    x2_out = x2[:,::-1] + 1

    Z = np.concatenate([x1_out, x2_out], 2)
    if sequence_lens is not str:
        for batch in range(len(sequence_lens)):
            # zero pad from the sequence_lens
            shape = list(np.shape(Z[batch]))
            seq_len = sequence_lens[batch]

            zero_pad = np.zeros([shape[0] - seq_len] + shape[1:])
            Z[batch] = np.concatenate([
                Z[batch][:seq_len], zero_pad])

    output = self._run_scan_node(initial, x1, x2, [6, 4], [3, 2],
                                 sequence_lens=sequence_lens,
                                 directions=directions)
    output_z = np.concatenate([output["z1"], output["z2"],
                               output["z3"], output["z4"]], 2)

    np.testing.assert_almost_equal(output["y"], Y)
    np.testing.assert_almost_equal(output_z, Z)

  def test_scan(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest(
          "ONNX version {} not supported.".format(
              defs.onnx_opset_version()))

    initial = self._get_rnd_int(0, 100, shape=[2]).astype(np.float32)
    x1 = self._get_rnd_float32(0, 1000, shape=[20, 6, 2])
    x2 = self._get_rnd_float32(0, 1000, shape=[20, 6, 2])

    Y = initial + np.shape(x1)[0]
    Z = np.concatenate([x1, x2], 1) + 1
 
    output = self._run_scan_node(initial, x1, x2, [6, 2], [3, 2])
    output_z = np.concatenate([output["z1"], output["z2"],
                               output["z3"], output["z4"]], 1)

    np.testing.assert_almost_equal(output["y"], Y)
    np.testing.assert_almost_equal(output_z, Z)

  def test_scan_input_directions(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest(
          "ONNX version {} not supported.".format(
              defs.onnx_opset_version()))

    initial = self._get_rnd_int(0, 100, shape=[1]).astype(np.float32)
    x1 = self._get_rnd_float32(0, 1000, shape=[20, 6, 2])
    x2 = self._get_rnd_float32(0, 1000, shape=[20, 6, 2])

    Y = initial + np.shape(x1)[0]
    Z = np.concatenate([x1[::-1], x2], 1) + 1

    output = self._run_scan_node(initial, x1, x2, [6, 2], [3, 2],
                                 scan_input_directions=[1, 0])
    output_z = np.concatenate([output["z1"], output["z2"],
                               output["z3"], output["z4"]], 1)

    np.testing.assert_almost_equal(output["y"], Y)
    np.testing.assert_almost_equal(output_z, Z)

  def test_scan_input_axes(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest(
          "ONNX version {} not supported.".format(
              defs.onnx_opset_version()))

    initial = self._get_rnd_int(0, 100, shape=[1]).astype(np.float32)
    x1 = self._get_rnd_float32(0, 1000, shape=[20, 6, 2])
    x2 = self._get_rnd_float32(0, 1000, shape=[20, 6, 2])

    Y = initial + np.shape(x1)[1]
    x1_transpose = np.transpose(x1, (1, 0, 2))
    x2_transpose = np.transpose(x2, (1, 0, 2))
    Z = np.concatenate([x1_transpose, x2_transpose], 1) + 1

    output = self._run_scan_node(initial, x1, x2, [3, 2], [10, 2],
                                 scan_input_axes=[1, 1])
    output_z = np.concatenate([output["z1"], output["z2"],
                               output["z3"], output["z4"]], 1)

    np.testing.assert_almost_equal(output["y"], Y)
    np.testing.assert_almost_equal(output_z, Z)

  def test_scan_output_directions(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest(
          "ONNX version {} not supported.".format(
              defs.onnx_opset_version()))

    initial = self._get_rnd_int(0, 100, shape=[1]).astype(np.float32)
    x1 = self._get_rnd_float32(0, 1000, shape=[20, 6, 2])
    x2 = self._get_rnd_float32(0, 1000, shape=[20, 6, 2])

    Y = initial + np.shape(x1)[0]
    Z = np.concatenate([x1, x2], 1) + 1

    output = self._run_scan_node(initial, x1, x2, [6, 2], [3, 2],
                                 scan_output_directions=[1, 0, 0, 1])
    output_z = np.concatenate([output["z1"][::-1], output["z2"],
                               output["z3"], output["z4"][::-1]], 1)

    np.testing.assert_almost_equal(output["y"], Y)
    np.testing.assert_almost_equal(output_z, Z)

  def test_scan_output_axes(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest(
          "ONNX version {} not supported.".format(
              defs.onnx_opset_version()))

    initial = self._get_rnd_int(0, 100, shape=[1]).astype(np.float32)
    x1 = self._get_rnd_float32(0, 1000, shape=[20, 6, 2])
    x2 = self._get_rnd_float32(0, 1000, shape=[20, 6, 2])

    Y = initial + np.shape(x1)[0]
    Z = np.concatenate([x1, x2], 1) + 1
    Z = np.transpose(Z, (1, 0, 2))

    output = self._run_scan_node(initial, x1, x2, [10, 2], [3, 2],
                                 scan_output_axes=[1, 1, 1, 1])
    output_z = np.concatenate([output["z1"], output["z2"],
                               output["z3"], output["z4"]], 0)

    np.testing.assert_almost_equal(output["y"], Y)
    np.testing.assert_almost_equal(output_z, Z)

  def test_shape(self):
    node_def = helper.make_node("Shape", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(output["Y"], np.shape(x))

  def test_shrink(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support Shrink.".format(
          defs.onnx_opset_version()))

    node_def = helper.make_node("Shrink", ["X"], ["Y"], bias=1.5, lambd=1.5)

    X = np.arange(-2.0, 2.1, dtype=np.float32)
    Y = np.array([-0.5, 0, 0, 0, 0.5], dtype=np.float32)
    output = run_node(node_def, [X])
    np.testing.assert_almost_equal(output["Y"], Y)

  def test_sigmoid(self):
    node_def = helper.make_node("Sigmoid", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[1000])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], 1 / (1 + np.exp(-x)))

  def test_sign(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support Sign.".format(
          defs.onnx_opset_version()))
    node_def = helper.make_node("Sign", ["X"], ["Y"])
    x = self._get_rnd_float32(-10, 10, [3, 5])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.sign(x))

  def test_sinh(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support Sinh.".format(
          defs.onnx_opset_version()))
    node_def = helper.make_node("Sinh", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[3, 4, 5])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.sinh(x))

  def test_size(self):
    node_def = helper.make_node("Size", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.size(x))

  def test_slice(self):
    # test case 1 with normal inputs
    axes = [0, 1, 2]
    starts = [0, 0, 0]
    ends = [2, 2, 2]
    steps = [1, 1, 1]

    if legacy_opset_pre_ver(10):
      node_def = helper.make_node("Slice", ["X"], ["S"],
                                  axes=axes,
                                  starts=starts,
                                  ends=ends)
      x = self._get_rnd_float32(shape=[1000]).reshape([10, 10, 10])
      output = run_node(node_def, [x])
      np.testing.assert_almost_equal(output["S"], x[0:2, 0:2, 0:2])
    else:
      node_def = helper.make_node("Slice",
                                  ["X", "starts", "ends", "axes", "steps"],
                                  ["S"])
      x = self._get_rnd_float32(shape=[1000]).reshape([10, 10, 10])
      output = run_node(node_def, [x, starts, ends, axes, steps])
      np.testing.assert_almost_equal(output["S"], x[0:2, 0:2, 0:2])

    # test case 2 with negative, out-of-bound and default inputs
    axes = [0, 2]
    starts = [0, -7]
    ends = [-8, 20]

    if legacy_opset_pre_ver(10):
      node_def = helper.make_node("Slice", ["X"], ["S"],
                                  axes=axes,
                                  starts=starts,
                                  ends=ends)
      x = self._get_rnd_float32(shape=[1000]).reshape([10, 10, 10])
      output = run_node(node_def, [x])
      np.testing.assert_almost_equal(output["S"], x[0:-8, :, -7:20])
    else:
      node_def = helper.make_node("Slice", ["X", "starts", "ends", "axes"],
                                  ["S"])
      x = self._get_rnd_float32(shape=[1000]).reshape([10, 10, 10])
      output = run_node(node_def, [x, starts, ends, axes])
      np.testing.assert_almost_equal(output["S"], x[0:-8, :, -7:20])

    # test case 3 with non-default steps
    axes = [0, 1, 2]
    starts = [0, 0, 0]
    ends = [2, 2, 2]
    steps = [2, -2, -1]

    if legacy_opset_pre_ver(10) == False:
      node_def = helper.make_node("Slice",
                                  ["X", "starts", "ends", "axes", "steps"],
                                  ["S"])
      x = self._get_rnd_float32(shape=[1000]).reshape([10, 10, 10])
      output = run_node(node_def, [x, starts, ends, axes, steps])
      np.testing.assert_almost_equal(output["S"], x[0:2:2, 0:2:-2, 0:2:-1])

  def test_softplus(self):
    node_def = helper.make_node("Softplus", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[3, 4, 5])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.log(np.exp(x) + 1))

  def test_softsign(self):
    node_def = helper.make_node("Softsign", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[3, 4, 5])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], x / (1 + np.abs(x)))

  def test_space_to_depth(self):
    node_def = helper.make_node("SpaceToDepth", ["X"], ["Y"], blocksize=2)
    x_shape = [1, 3, 2, 2]
    x = self._get_rnd_float32(shape=x_shape)
    output = run_node(node_def, [x])
    x = np.transpose(x, (0, 2, 3, 1))
    y = np.reshape(np.swapaxes(x.reshape(1, 1, 1, 1, 1, 12), 2, 3),
                   (1, 1, 1, 12))
    y = np.transpose(y, (0, 3, 1, 2))
    np.testing.assert_allclose(output["Y"], y, rtol=1e-3)

  def test_split(self):
    split = [3, 3, 4]
    node_def = helper.make_node("Split", ["X"],
                                ["Z%i" % i for i in range(len(split))],
                                axis=0,
                                split=split)
    x = self._get_rnd_float32(shape=[100]).reshape([10, 10])

    output = run_node(node_def, [x])
    for a, b in zip(list(output), np.split(x, np.cumsum(split))[:-1]):
      np.testing.assert_almost_equal(a, b)

  def test_sqrt(self):
    node_def = helper.make_node("Sqrt", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[1000]) + 1.0
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.sqrt(x), decimal=5)

  def test_squeeze(self):
    node_def = helper.make_node("Squeeze", ["X"], ["Y"], axes=[2])
    x = np.array([[[0], [1], [2]]])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.squeeze(x, axis=2))

  def test_sub(self):
    node_def = helper.make_node("Sub", ["X", "Y"], ["Z"])
    x = self._get_rnd_float32(shape=[10, 10])
    y = self._get_rnd_float32(shape=[10, 10])
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.subtract(x, y))

  def test_sum(self):
    node_def = helper.make_node("Sum", ["X1", "X2", "X3", "X4"], ["Z"])
    x1 = self._get_rnd_float32(shape=[10, 10])
    x2 = self._get_rnd_float32(shape=[10, 10])
    x3 = self._get_rnd_float32(shape=[10, 10])
    x4 = self._get_rnd_float32(shape=[10, 10])
    output = run_node(node_def, [x1, x2, x3, x4])
    test_output = x1 + x2 + x3 + x4
    np.testing.assert_almost_equal(output["Z"], test_output)

  def test_tanh(self):
    node_def = helper.make_node("Tanh", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[1000]) + 1.0
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.tanh(x), decimal=5)

  def test_tfidf_vectorizer(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support TfIdfVectorizer.".format(
                   defs.onnx_opset_version()))

    def run_test_ints():
      node_def = helper.make_node("TfIdfVectorizer", ["X"], ["Y"],
                   mode = mode, min_gram_length=min_gram_len, max_gram_length=max_gram_len,
                   max_skip_count=max_skip, ngram_counts=ngram_counts,
                   ngram_indexes=ngram_indexes, weights=weights, pool_int64s=pool_int64s)
      output = run_node(node_def, [x])
      np.testing.assert_almost_equal(output["Y"], y)
    def run_test_strings():
      node_def =  helper.make_node("TfIdfVectorizer", ["X"], ["Y"],
                   mode = mode, min_gram_length=min_gram_len, max_gram_length=max_gram_len,
                   max_skip_count=max_skip, ngram_counts=ngram_counts,
                   ngram_indexes=ngram_indexes, weights=weights, pool_strings=pool_strings)
      output = run_node(node_def, [x])
      np.testing.assert_almost_equal(output["Y"], y)

    # test 2d inputs with 3 elements, output contains 1-grams and 2-grams
    x = np.array([[1, 1, 3, 3, 3, 7], [8, 6, 7, 5, 6, 8], [8, 6, 7, 5, 6, 8]]).astype(np.int32)
    y = np.array([[0., 3., 0., 0., 0., 0., 0.], [0., 0., 1., 0., 1., 0., 1.], [0., 0., 1., 0., 1., 0., 1.]]).astype(np.float32)
    ngram_counts = np.array([0, 4]).astype(np.int64)
    ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
    pool_int64s = np.array([2, 3, 5, 4, 5, 6, 7, 8, 6, 7]).astype(np.int64)
    min_gram_len = 1
    max_gram_len = 2
    max_skip = 0
    mode = 'TF'
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    run_test_ints()

    # test 1d inputs with indexes in non-default order, max_skip=3, output 2-grams
    x = np.array([1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8]).astype(np.int32)
    y = np.array([0., 1., 0., 1., 0., 0., 2.]).astype(np.float32)
    ngram_counts = np.array([0, 4]).astype(np.int64)
    ngram_indexes = np.array([5, 0, 2, 4, 1, 6, 3]).astype(np.int64)
    pool_int64s = np.array([2, 3, 5, 4, 5, 6, 7, 8, 6, 7]).astype(np.int64)
    min_gram_len = 2
    max_gram_len = 2
    max_skip = 3
    mode = 'TF'
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    run_test_ints()

    # test IDF mode with weights, max_skip=5, output contains 1-grams and 2-grams
    x = np.array([[1, 1, 3, 3, 3, 7], [8, 6, 7, 5, 6, 8]]).astype(np.int32)
    y = np.array([[0., 0.1, 0., 0., 0., 0., 0.], [0., 0., 0.1, 0., 0.5, 0.5, 0.5]]).astype(np.float32)
    ngram_counts = np.array([0, 4]).astype(np.int64)
    ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
    pool_int64s = np.array([2, 3, 5, 4, 5, 6, 7, 8, 6, 7]).astype(np.int64)
    min_gram_len = 1
    max_gram_len = 2
    max_skip = 5
    mode = 'IDF'
    weights = np.array([0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
    run_test_ints()

    # test strings inputs, max_skip=5, output contains 1-grams and 2-grams
    x = np.array(['a', 'a', 'b', 'b', 'b', 'c', 'd', 'e', 'c', 'f', 'e', 'd'])
    y = np.array([0., 3., 1., 0., 1., 3., 1.]).astype(np.float32)
    ngram_counts = np.array([0, 4]).astype(np.int64)
    ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
    pool_strings = np.array(['x', 'b', 'f', 'y', 'f', 'e', 'c', 'd', 'e', 'c'])
    min_gram_len = 1
    max_gram_len = 2
    max_skip = 5
    mode = 'TF'
    run_test_strings()

  def test_thresholded_relu(self):
    alpha = 2.0
    node_def = helper.make_node("ThresholdedRelu", ["X"], ["Y"], alpha=alpha)
    x = self._get_rnd_float32(-3.0, 3.0, [10])
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
    x = self._get_rnd_float32(shape=[3, 5, 5, 3])
    repeats = [1, 1, 2, 1]
    output = run_node(node_def, [x, repeats])
    np.testing.assert_allclose(output["Z"], np.tile(x, repeats), rtol=1e-3)

  def test_transpose(self):
    node_def = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 1])
    x = self._get_rnd_float32(shape=[1000]).reshape([10, 10, 10])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.transpose(x, (0, 2, 1)))

  def test_topk(self):
    x = np.arange(15, dtype=np.float32).reshape(3, 5)
    values = np.array([[4, 3], [9, 8], [14, 13]], dtype=np.float32)
    indices = np.array([[4, 3], [4, 3], [4, 3]], dtype=np.int64)
    if legacy_opset_pre_ver(10):  # for opset = 1
      node_def = helper.make_node("TopK", ["x"], ["values", "indices"], k=2)
      output = run_node(node_def, [x])
    elif legacy_opset_pre_ver(11):  # for opset = 10
      k = np.array([2], dtype=np.int64)
      node_def = helper.make_node("TopK", ["x", "k"], ["values", "indices"])
      output = run_node(node_def, [x, k])
    else:  # for opset = 11
      x = np.array([[3, 2, 5, 10, 7], [12, 15, 10, 7, 20], [21, 16, 5, 3, 6]],
                   dtype=np.float32)
      values = np.array([[3, 2], [10, 7], [5, 3]], dtype=np.float32)
      indices = np.array([[0, 1], [2, 3], [2, 3]], dtype=np.int64)
      k = np.array([2], dtype=np.int64)
      node_def = helper.make_node("TopK", ["x", "k"], ["values", "indices"],
                                  largest=0,
                                  sorted=0)
      output = run_node(node_def, [x, k])
    np.testing.assert_almost_equal(output["values"], values)
    np.testing.assert_almost_equal(output["indices"], indices)

  def test_where(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support Where.".format(
          defs.onnx_opset_version()))
    node_def = helper.make_node("Where", ["C", "X", "Y"], ["Z"])
    c = np.array([[1, 0], [1, 1]], dtype=np.bool)
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[9, 8], [7, 6]], dtype=np.float32)
    output = run_node(node_def, [c, x, y])
    np.testing.assert_almost_equal(output["Z"], np.where(c, x, y))

  def test_scatter_nd(self):
    if legacy_opset_pre_ver(11):
      raise unittest.SkipTest("ONNX version {} doesn't support ScatterND.".format(
          defs.onnx_opset_version()))
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    indices = np.array([[4], [3], [1], [7]], dtype=np.int64)
    updates = np.array([9, 10, 11, 12], dtype=np.float32)
    ref_output = np.array([1, 11, 3, 10, 9, 6, 7, 12], dtype=np.float32)
    node_def = helper.make_node("ScatterND", ["data", "indices", "updates"],
                                ["outputs"])
    output = run_node(node_def, [data, indices, updates])
    np.testing.assert_almost_equal(output["outputs"], ref_output)

  def test_scatter_elements1(self):
    data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    indices = np.array([[1, 3]], dtype=np.int64)
    updates = np.array([[1.1, 2.1]], dtype=np.float32)
    axis = 1
    ref_output = np.array([[1.0, 1.1, 3.0, 2.1, 5.0]], dtype=np.float32)

    if legacy_opset_pre_ver(11):
      node_def = helper.make_node("Scatter",
                                ["data", "indices", "updates"], ["outputs"],
                                axis=axis)
      output = run_node(node_def, [data, indices, updates])
      np.testing.assert_almost_equal(output["outputs"], ref_output)
    else:
      node_def = helper.make_node("ScatterElements",
                                ["data", "indices", "updates"], ["outputs"],
                                axis=axis)
      output = run_node(node_def, [data, indices, updates])
      np.testing.assert_almost_equal(output["outputs"], ref_output)

  def test_scatter_elements2(self):
    data = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)
    indices = np.array([
        [1, 0, 2],
        [0, 2, 1],
    ], dtype=np.int64)
    updates = np.array([
        [1.0, 1.1, 1.2],
        [2.0, 2.1, 2.2],
    ], dtype=np.float32)
    ref_output = np.array([
        [2.0, 1.1, 0.0],
        [1.0, 0.0, 2.2],
        [0.0, 2.1, 1.2],
    ], dtype=np.float32)

    if legacy_opset_pre_ver(11):
      node_def = helper.make_node("Scatter",
                                  ["data", "indices", "updates"], ["outputs"])
      output = run_node(node_def, [data, indices, updates])
      np.testing.assert_almost_equal(output["outputs"], ref_output)
    else:
      node_def = helper.make_node("ScatterElements",
                                ["data", "indices", "updates"], ["outputs"])
      output = run_node(node_def, [data, indices, updates])
      np.testing.assert_almost_equal(output["outputs"], ref_output)

  def test_gather_nd(self):
    if legacy_opset_pre_ver(11):
      raise unittest.SkipTest("ONNX version {} doesn't support GatherND.".format(
          defs.onnx_opset_version()))
    data = [[0, 1], [2, 3]]
    indices = [[0, 0], [1, 1]]
    ref_output = [0, 3]
    node_def = helper.make_node("GatherND", ["data", "indices"], ["outputs"])
    output = run_node(node_def, [data, indices])
    np.testing.assert_almost_equal(output["outputs"], ref_output)


if __name__ == '__main__':
  unittest.main()
