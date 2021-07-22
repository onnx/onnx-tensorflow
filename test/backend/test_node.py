from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import unittest

from onnx import defs
from onnx import helper
from onnx import TensorProto
from onnx.backend.test.case.node import hardmax
from onnx.backend.test.case.node.onehot import one_hot
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from onnx_tf.backend import onnx_graph_to_tensorflow_rep
from onnx_tf.backend import run_node
from onnx_tf.common import supports_device
from onnx_tf.common.legacy import legacy_onnx_pre_ver, legacy_opset_pre_ver
from onnx_tf.common.pooling_helper import py_pool


class TestNode(unittest.TestCase):
  """ Tests for nodes
  """

  def _get_device_list(self):
    # Check does the environment support CUDA.
    return ['CPU', 'CUDA'] if supports_device("CUDA") else ['CPU']

  def _get_rnd_float32(self, low=-1.0, high=1.0, shape=None):
    output = np.random.uniform(low, high, shape)
    if shape is None:
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
    x = self._get_rnd_float32(low=1.0,
                              high=np.finfo(np.float32).max,
                              shape=[3, 4, 5])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.arccosh(x), decimal=5)

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

      # test select_last_index
      if not legacy_opset_pre_ver(12):
        # select_last_index = 0
        node_def = helper.make_node("ArgMax", ["data"], ["reduced"],
                                    axis=axis,
                                    keepdims=0,
                                    select_last_index=0)
        data = self._get_rnd_float32(shape=[10, 10])
        output = run_node(node_def, [data])
        np.testing.assert_almost_equal(output["reduced"],
                                       np.argmax(data, axis=axis))
        # select_last_index = 1
        node_def = helper.make_node("ArgMax", ["data"], ["reduced"],
                                    axis=axis,
                                    keepdims=0,
                                    select_last_index=1)
        data = np.array([[1, 2, 3, 5, 3, 4, 5, 1], [2, 9, 3, 5, 9, 4, 5, 1]])
        output = run_node(node_def, [data])
        data = np.flip(data, axis)
        result = np.argmax(data, axis=axis)
        result = data.shape[axis] - result - 1
        np.testing.assert_almost_equal(output["reduced"], result)

  def test_arg_min(self):
    for axis in [0, 1]:
      node_def = helper.make_node("ArgMin", ["data"], ["reduced"],
                                  axis=axis,
                                  keepdims=0)
      data = self._get_rnd_float32(shape=[10, 10])
      output = run_node(node_def, [data])
      np.testing.assert_almost_equal(output["reduced"],
                                     np.argmin(data, axis=axis))

      # test select_last_index
      if not legacy_opset_pre_ver(12):
        # select_last_index = 0
        node_def = helper.make_node("ArgMin", ["data"], ["reduced"],
                                    axis=axis,
                                    keepdims=0,
                                    select_last_index=0)
        data = self._get_rnd_float32(shape=[10, 10])
        output = run_node(node_def, [data])
        np.testing.assert_almost_equal(output["reduced"],
                                       np.argmin(data, axis=axis))
        # select_last_index = 1
        node_def = helper.make_node("ArgMin", ["data"], ["reduced"],
                                    axis=axis,
                                    keepdims=0,
                                    select_last_index=1)
        data = np.array([[1, 2, 3, 5, 3, 4, 5, 1], [2, 7, 3, 5, 2, 4, 5, 6]])
        output = run_node(node_def, [data])
        data = np.flip(data, axis)
        result = np.argmin(data, axis=axis)
        result = data.shape[axis] - result - 1
        np.testing.assert_almost_equal(output["reduced"], result)

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
    np.testing.assert_almost_equal(output["Y"], np.arctanh(x), decimal=6)

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
      # added casting to bfloat16 from number in opset 13
      if not legacy_opset_pre_ver(13):
        test_cases.append((TensorProto.BFLOAT16, tf.bfloat16))
    for ty, tf_type in test_cases:
      node_def = helper.make_node("Cast", ["input"], ["output"], to=ty)
      vector = np.array([2, 3])
      output = run_node(node_def, [vector])
      np.testing.assert_equal(output["output"].dtype, tf_type)
    if not legacy_opset_pre_ver(9):
      # test_cases2 is focused on Strings to Numbers
      # Note: casting from string to bfloat16 is not allowed by tf.strings.to_number
      # so no BFLOAT16 in test_cases2.
      test_cases2 = [(TensorProto.FLOAT, tf.float32),
                     (TensorProto.INT32, tf.int32),
                     (TensorProto.INT64, tf.int64),
                     (TensorProto.DOUBLE, tf.float64)]
      for ty, tf_type in test_cases2:
        node_def = helper.make_node("Cast", ["input"], ["output"], to=ty)
        vector = np.array(['2', '3'])
        output = run_node(node_def, [vector])
        np.testing.assert_equal(output["output"].dtype, tf_type)

    if not legacy_opset_pre_ver(9):
      # test_case3 is focused on Strings to float and the special floating-point values.
      test_cases3 = [(TensorProto.FLOAT, tf.float32),
                     (TensorProto.DOUBLE, tf.float64)]
      for ty, tf_type in test_cases3:
        node_def = helper.make_node("Cast", ["input"], ["output"], to=ty)
        vector = np.array(['3.14159', '1e-5', '1E8', 'NaN', '-INF', '+INF'])
        output = run_node(node_def, [vector])
        np.testing.assert_equal(output["output"].dtype, tf_type)

  def test_ceil(self):
    node_def = helper.make_node("Ceil", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[1000])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.ceil(x))

  def test_celu(self):
    if legacy_opset_pre_ver(12):
      raise unittest.SkipTest("ONNX version {} doesn't support Celu.".format(
          defs.onnx_opset_version()))
    alpha = 2.0
    node_def = helper.make_node("Celu", ["X"], ["Y"], alpha=alpha)
    x = np.array([[[-1.0763247, 0.98948643, 0.22292195],
                   [0.1751388, -1.39814249, 1.44396422]]],
                 dtype=np.float32)
    output = run_node(node_def, [x])
    positive_input = np.maximum(0, x)
    negative_input = np.minimum(0, alpha * (np.exp(x / alpha) - 1))
    expected_output = positive_input + negative_input
    np.testing.assert_almost_equal(output["Y"], expected_output)

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
      b = tf.sparse.SparseTensor(output["Y"].indices, output["Y"].values,
                                 output["Y"].dense_shape)
      b = tf.sparse.to_dense(b)
      result = b.numpy()
      np.testing.assert_equal(result, expected)

    if not legacy_opset_pre_ver(12):
      float_attr = 1.0
      floats_attr = [1.0, 2.0, 3.0]
      int_attr = np.int64(123)
      ints_attr = [np.int64(4), np.int64(5), np.int64(6)]
      string_attr = 'The Cat in the Hat'
      strings_attr = [
          'Green Eggs and Ham', 'How the Grinch Stole Christmas!',
          'The Cat in the Hat Comes Back'
      ]
      testcases = [(helper.make_node("Constant", [], ["Y"],
                                     value_float=float_attr), float_attr),
                   (helper.make_node("Constant", [], ["Y"],
                                     value_floats=floats_attr), floats_attr),
                   (helper.make_node("Constant", [], ["Y"],
                                     value_int=int_attr), int_attr),
                   (helper.make_node("Constant", [], ["Y"],
                                     value_ints=ints_attr), ints_attr),
                   (helper.make_node("Constant", [], ["Y"],
                                     value_string=string_attr), string_attr),
                   (helper.make_node("Constant", [], ["Y"],
                                     value_strings=strings_attr), strings_attr)]
      for node_def, expected in testcases:
        output = run_node(node_def, [])
        if isinstance(expected, str):
          np.testing.assert_string_equal(output["Y"].decode('UTF-8'), expected)
        elif isinstance(expected, list) and isinstance(expected[0], str):
          for i in range(len(expected)):
            np.testing.assert_string_equal(output['Y'][i].decode('UTF-8'),
                                           expected[i])
        else:
          np.testing.assert_equal(output["Y"], expected)

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
    for device in self._get_device_list():
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

      np.testing.assert_almost_equal(output["Y"], test_output, decimal=4)

  def test_conv_integer(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support ConvInteger.".format(
              defs.onnx_opset_version()))

    for device in self._get_device_list():
      # Test w_zero_point
      x = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.int8).reshape(
          (1, 1, 3, 3))
      w = np.array([2, 2, 2, 2]).astype(np.int8).reshape((1, 1, 2, 2))
      w_zero_point = np.int8(1)
      y = np.array([16, 20, 28, 32]).astype(np.int32).reshape((1, 1, 2, 2))

      node = helper.make_node("ConvInteger",
                              ["X", "W", "x_zero_point", "w_zero_point"], ["Y"],
                              kernel_shape=[2, 2],
                              pads=[0, 0, 0, 0],
                              dilations=[1, 1])
      output = run_node(node, [x, w, np.int8(0), w_zero_point], device=device)
      np.testing.assert_almost_equal(output["Y"], y)

      # Test x_zero_point and w_zero_point
      x = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.int8).reshape(
          (1, 1, 3, 3))
      x_zero_point = np.int8(1)
      w = np.array([2, 2, 2, 2]).astype(np.int8).reshape((1, 1, 2, 2))
      w_zero_point = np.int8(1)
      y = np.array([12, 16, 24, 28]).astype(np.int32).reshape((1, 1, 2, 2))

      node = helper.make_node("ConvInteger",
                              ["X", "W", "x_zero_point", "w_zero_point"], ["Y"],
                              kernel_shape=[2, 2],
                              pads=[0, 0, 0, 0],
                              dilations=[1, 1])
      output = run_node(node, [x, w, x_zero_point, w_zero_point], device=device)
      np.testing.assert_almost_equal(output["Y"], y)

      # Test w_zero_point as 1d tensor
      x = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.int8).reshape(
          (1, 1, 3, 3))
      w = np.array([2, 2, 2, 2]).astype(np.int8).reshape((1, 1, 2, 2))
      w_zero_point = np.array([1]).astype(np.int8)
      y = np.array([16, 20, 28, 32]).astype(np.int32).reshape((1, 1, 2, 2))

      node = helper.make_node("ConvInteger",
                              ["X", "W", "x_zero_point", "w_zero_point"], ["Y"],
                              kernel_shape=[2, 2],
                              pads=[0, 0, 0, 0],
                              dilations=[1, 1])
      output = run_node(node, [x, w, np.int8(0), w_zero_point], device=device)
      np.testing.assert_almost_equal(output["Y"], y)

      # Test w_zero_point as 1d tensor shape 2
      x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(np.int8).reshape(
          (1, 1, 3, 3))
      w = np.array([2, 2, 2, 2, 2, 2, 2, 2]).astype(np.int8).reshape(
          (2, 1, 2, 2))
      w_zero_point = np.array([1, 2]).astype(np.int8)
      y = np.array([12, 16, 24, 28, 0, 0, 0, 0]).astype(np.int32).reshape(
          (1, 2, 2, 2))

      node = helper.make_node("ConvInteger",
                              ["X", "W", "x_zero_point", "w_zero_point"], ["Y"],
                              kernel_shape=[2, 2],
                              pads=[0, 0, 0, 0],
                              dilations=[1, 1])
      output = run_node(node, [x, w, np.int8(0), w_zero_point], device=device)
      np.testing.assert_almost_equal(output["Y"], y)

  def test_conv_transpose(self):
    for device in self._get_device_list():
      pads = [1, 1]
      node_def = helper.make_node("ConvTranspose", ["X", "weights"], ["Y"],
                                  pads=pads)
      x_shape = [1, 3, 4]
      x = self._get_rnd_float32(shape=x_shape)
      weight_shape = [3, 5, 2]
      weights = self._get_rnd_float32(shape=weight_shape)
      output = run_node(node_def, [x, weights], device=device)

      padh_left = weight_shape[2] - 1 - pads[0]
      padh_right = weight_shape[2] - 1 - pads[1]
      kh = weight_shape[2]
      outh = x_shape[2] + padh_right + padh_right - (kh - 1)

      out_shape = [x_shape[0], weight_shape[1], outh]

      test_output = np.zeros(out_shape)
      for b in range(0, x_shape[0]):
        for m in range(0, weight_shape[1]):
          for c in range(0, x_shape[1]):
            for h in range(0, outh):
              for k in range(h, h + kh):
                if (k - padh_left >= 0):
                  test_output[b][m][h] += x[b][c][
                      k - padh_left] * weights[c][m][kh + h - 1 - k]

      np.testing.assert_almost_equal(output["Y"], test_output, decimal=5)

      # test for spatial dimension of colnolution is 2
      pads = [1, 1, 1, 1]
      node_def = helper.make_node("ConvTranspose", ["X", "weights"], ["Y"],
                                  pads=pads)
      x_shape = [1, 3, 4, 6]
      x = self._get_rnd_float32(shape=x_shape)
      weight_shape = [3, 5, 2, 2]
      weights = self._get_rnd_float32(shape=weight_shape)
      output = run_node(node_def, [x, weights], device=device)

      padh_left = weight_shape[2] - 1 - pads[0]
      padh_right = weight_shape[2] - 1 - pads[1]
      padw_left = weight_shape[3] - 1 - pads[2]
      padw_right = weight_shape[3] - 1 - pads[3]

      kh = weight_shape[2]
      kw = weight_shape[3]
      outh = x_shape[2] + padh_right + padh_right - (kh - 1)
      outw = x_shape[3] + padw_right + padw_right - (kw - 1)

      out_shape = [x_shape[0], weight_shape[1], outh, outw]

      test_output = np.zeros(out_shape)
      for b in range(0, x_shape[0]):
        for m in range(0, weight_shape[1]):
          for c in range(0, x_shape[1]):
            for h in range(0, outh):
              for w in range(0, outw):
                for k1 in range(h, h + kh):
                  for k2 in range(w, w + kw):
                    if (k1 - padh_left >= 0 and k2 - padw_left >= 0):
                      test_output[b][m][h][w] += x[b][c][k1 - padh_left][
                          k2 - padw_left] * weights[c][m][kh + h - 1 -
                                                          k1][kw + w - 1 - k2]

      np.testing.assert_almost_equal(output["Y"], test_output, decimal=5)

  def test_cosh(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support Cosh.".format(
          defs.onnx_opset_version()))
    node_def = helper.make_node("Cosh", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[3, 4, 5])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.cosh(x))

  def test_cumsum(self):
    if legacy_opset_pre_ver(11):
      raise unittest.SkipTest("ONNX version {} doesn't support CumSum.".format(
          defs.onnx_opset_version()))
    x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int32)
    axis = 0
    node_def = helper.make_node("CumSum", ["x", "axis"], ["y"])
    # note: if axis is not provided, np.cumsum() will compute over flattened array,
    # which is different than the TensorFlow behavior
    y = np.cumsum(x, axis).astype(np.int32)
    output = run_node(node_def, [x, axis])
    np.testing.assert_almost_equal(output["y"], y)
    # test data types that are not natively supported by Tensorflow
    x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.uint32)
    y = np.cumsum(x, axis).astype(np.uint32)
    output = run_node(node_def, [x, axis])
    np.testing.assert_almost_equal(output["y"], y)

  def test_depth_to_space(self):
    for device in self._get_device_list():
      node_def = helper.make_node("DepthToSpace", ["X"], ["Y"], blocksize=2)
      x_shape = [1, 12, 1, 1]
      x = self._get_rnd_float32(shape=x_shape)
      output = run_node(node_def, [x], device=device)
      x = np.transpose(x, (0, 2, 3, 1))
      y = np.reshape(np.swapaxes(x.reshape(1, 1, 1, 2, 2, 3), 2, 3),
                     (1, 2, 2, 3))
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
    # in ONNX. This has changed in Opset 12. In Opset
    # 12 ONNX has added support for training.

    x = self._get_rnd_float32(shape=[3, 4, 5])
    if legacy_opset_pre_ver(7):
      # at inference mode, is_test attribute is always set to 1
      node_def = helper.make_node("Dropout", ["X"], ["Y"], is_test=1)
      y = x  # output is same is input i.e. nothing is dropped
      output = run_node(node_def, [x])
      np.testing.assert_equal(output["Y"], y)
    elif legacy_opset_pre_ver(12):  # for Opset 7, 10
      # no is_test attribute anymore
      node_def = helper.make_node("Dropout", ["X"], ["Y"])
      y = x  # output is same is input i.e. nothing is dropped
      output = run_node(node_def, [x])
      np.testing.assert_equal(output["Y"], y)
    else:  # for Opset 12, 13
      # Inference mode tests
      #   training_mode is false by default
      #   ratio is ignored
      #   nothing will be dropped from the input data
      #   if mask is requested as output it will contain all ones

      # Inference, mask not requested
      node_def = helper.make_node("Dropout", ["X"], ["Y"])
      y = x  # output is same is input i.e. nothing is dropped
      output = run_node(node_def, [x])
      np.testing.assert_equal(output["Y"], y)

      # Inference, mask requested
      node_def = helper.make_node("Dropout", inputs=["X"], outputs=["Y", "Z"])
      y = x  # output is same is input i.e. nothing is dropped
      z = np.ones(x.shape, dtype=bool)  # mask returned is all ones
      output = run_node(node_def, [x])
      np.testing.assert_equal(output["Y"], y)
      np.testing.assert_equal(output["Z"], z)

      # Training mode tests
      #   training_mode is true
      #   output is  a random dropout that scales masked
      #     input using the following equation
      #     output = scale * data * mask
      #     scale = 1. / (1. - ratio)

      ratio = np.float32(0.5)
      training_mode = np.bool_(True)
      no_of_runs = 20  # run 20 times and make sure that on average we have the desired dropout

      # Training, mask not requested
      node_def = helper.make_node("Dropout",
                                  inputs=["X", "X1", "X2"],
                                  outputs=["Y"])
      sum_ratio_in_output = 0
      for _ in range(no_of_runs):
        output = run_node(node_def, [x, ratio, training_mode])
        output_nonzero_count = np.count_nonzero(output["Y"])
        output_size = output["Y"].size
        ratio_in_output = (output_size - output_nonzero_count) / output_size
        sum_ratio_in_output += ratio_in_output
      ratio_in_output = sum_ratio_in_output / no_of_runs
      # test by confirming that the dropout is close to the ratio passed in
      np.testing.assert_almost_equal(ratio_in_output, ratio, decimal=1)

      # Training, mask requested
      node_def = helper.make_node("Dropout",
                                  inputs=["X", "X1", "X2"],
                                  outputs=["Y", "Z"])
      sum_ratio_in_output = 0
      for _ in range(no_of_runs):
        output = run_node(node_def, [x, ratio, training_mode])
        output_nonzero_count = np.count_nonzero(output["Y"])
        output_size = output["Y"].size
        ratio_in_output = (output_size - output_nonzero_count) / output_size
        sum_ratio_in_output += ratio_in_output
      ratio_in_output = sum_ratio_in_output / no_of_runs
      # test by confirming that the dropout is close to the ratio passed in
      np.testing.assert_almost_equal(ratio_in_output, ratio, decimal=1)
      # test mask
      np.testing.assert_equal(output["Z"], output["Y"].astype(bool))

  def test_dot(self):
    # this op is removed
    # remove this test in the future
    return
    node_def = helper.make_node("Dot", ["X", "Y"], ["Z"])
    x = np.floor(self._get_rnd_float32(shape=[10, 10]))
    y = np.floor(self._get_rnd_float32(shape=[10, 10]))
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.dot(x, y))

  def test_dynamic_quantize_linear(self):
    if legacy_opset_pre_ver(11):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support DynamicQuantizeLinear.".format(
              defs.onnx_opset_version()))
    node_def = helper.make_node("DynamicQuantizeLinear", ["X"],
                                ["Y", "Y_Scale", "Y_Zero_Point"])
    x = self._get_rnd_float32(shape=[3, 4])
    min_x = np.minimum(0, np.min(x))
    max_x = np.maximum(0, np.max(x))
    y_scale = np.float32((max_x - min_x) / (255 - 0))  # uint8 -> [0, 255]
    y_zero_point = np.clip(round((0 - min_x) / y_scale), 0,
                           255).astype(np.uint8)
    y = np.clip(np.round(x / y_scale) + y_zero_point, 0, 255).astype(np.uint8)
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], y)
    np.testing.assert_almost_equal(output["Y_Scale"], y_scale)
    np.testing.assert_almost_equal(output["Y_Zero_Point"], y_zero_point)

  def test_einsum(self):
    if legacy_opset_pre_ver(12):
      raise unittest.SkipTest("ONNX version {} doesn't support Einsum.".format(
          defs.onnx_opset_version()))
    equation = 'ij,jk->ik'  #matmul
    node_def = helper.make_node("Einsum", ["X", "Y"], ["Z"], equation=equation)
    x = self._get_rnd_float32(shape=[3, 4])
    y = self._get_rnd_float32(shape=[4, 5])
    z = np.einsum(equation, x, y)
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], z)

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
    # test data types that are not natively supported by Tensorflow
    x = np.arange(8).reshape((2, 2, 2)).astype(np.uint16)
    y = np.arange(8).reshape((2, 2, 2)).astype(np.uint16)
    output = run_node(node_def, [x, y])
    np.testing.assert_equal(output["Z"], np.equal(x, y))

    x = np.arange(8).reshape((2, 2, 2)).astype(np.uint64)
    y = np.arange(8).reshape((2, 2, 2)).astype(np.uint64)
    self.assertRaises(RuntimeError, run_node, node_def, [x, y])

  def test_erf(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support Erf.".format(
          defs.onnx_opset_version()))
    node_def = helper.make_node("Erf", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[3, 4, 5])
    output = run_node(node_def, [x])
    exp_output = np.vectorize(math.erf)(x).astype(np.float32)
    np.testing.assert_allclose(output['Y'], exp_output, rtol=1e-6, atol=1e-6)

  def test_exp(self):
    node_def = helper.make_node("Exp", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[100])
    x = x - 3.6
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.exp(x))

  def test_expand(self):
    node_def = helper.make_node("Expand", ["X", "shape"], ["Y"])
    x = np.array([[True], [False], [True]])
    shape = [2, 1, 6]
    y = x * np.ones(shape, dtype=np.bool)
    output = run_node(node_def, [x, shape])
    np.testing.assert_almost_equal(output["Y"], y)

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
    y = np.array([[0, 1], [1, 2]])
    output = run_node(node_def, [x, y])
    test_output = np.zeros((2, 2, 10))
    for i in range(0, 2):
      for j in range(0, 10):
        test_output[0][i][j] = x[i][j]
    for i in range(0, 2):
      for j in range(0, 10):
        test_output[1][i][j] = x[i + 1][j]
    np.testing.assert_almost_equal(output["Z"], test_output)
    if defs.onnx_opset_version() >= 11:
      # test negative indices
      y = np.array([[-10, -9], [1, -8]])
      output = run_node(node_def, [x, y])
      np.testing.assert_almost_equal(output["Z"], test_output)
      # test out of bound indices
      for y in (np.array([[-10, 11], [1, -8]]), np.array([[-10, -11], [1, -8]])):
        try:
          output = run_node(node_def, [x, y])
          np.testing.assert_almost_equal(output["Z"], test_output)
          raise AssertionError("Expected ValueError not raised for indices %d" %
                               str(y))
        except tf.errors.InvalidArgumentError as e:
          assert 'Gather indices are out of bound' in str(e), str(y)
      # test non-0 and negative axis
      axis = -3
      node_def = helper.make_node("Gather", ["X", "Y"], ["Z"], axis=axis)
      x = np.reshape(np.arange(5 * 4 * 3 * 2), (5, 4, 3, 2))
      y = np.array([0, 1, 3])
      test_output = np.take(x, y, axis=axis)
      output = run_node(node_def, [x, y])
      np.testing.assert_almost_equal(output["Z"], test_output)
      # test axis attribute validation
      for axis in [-5, 4, 10]:
        try:
          node_def = helper.make_node("Gather", ["X", "Y"], ["Z"], axis=axis)
          run_node(node_def, [x, y])
          raise AssertionError(
              "Expected ValueError not raised for axis value %d" % axis)
        except ValueError as e:
          assert 'out of bounds' in str(e), str(e) + ' for axis ' + str(axis)

  def test_gather_elements(self):
    if legacy_opset_pre_ver(11):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support GatherElements.".format(
              defs.onnx_opset_version()))

    data_dtype = np.int32
    data = np.array([
        [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]],
        [[20, 21], [22, 23], [24, 25], [26, 27], [28, 29]],
        [[30, 31], [32, 33], [34, 35], [36, 37], [38, 39]],
        [[40, 41], [42, 43], [44, 45], [46, 47], [48, 49]],
    ],
                    dtype=data_dtype)

    # test default axis
    indices = np.array([[[3, 0], [2, 0], [1, 0], [0, 0], [3, 1]]],
                       dtype=np.int64)
    ref_output = np.array([[[40, 11], [32, 13], [24, 15], [16, 17], [48, 29]]],
                          dtype=data_dtype)
    node_def = helper.make_node("GatherElements", ["data", "indices"],
                                ["outputs"])
    output = run_node(node_def, [data, indices])
    np.testing.assert_almost_equal(output["outputs"], ref_output)

    # test non-default axis
    indices = np.array([[[3, 0], [2, 1], [1, 2], [0, 3]]], dtype=data_dtype)
    ref_output = np.array([
        [[16, 11], [14, 13], [12, 15], [10, 17]],
    ],
                          dtype=data_dtype)
    node_def = helper.make_node("GatherElements", ["data", "indices"],
                                ["outputs"],
                                axis=1)
    output = run_node(node_def, [data, indices])
    np.testing.assert_almost_equal(output["outputs"], ref_output)

    # test negative axis
    indices = np.array([
        [[1, 1, -2], [1, -2, 1], [-1, 1, -1], [-2, 1, -2], [-2, 1, 1]],
    ],
                       dtype=data_dtype)
    ref_output = np.array([
        [[11, 11, 10], [13, 12, 13], [15, 15, 15], [16, 17, 16], [18, 19, 19]],
    ],
                          dtype=data_dtype)
    node_def = helper.make_node("GatherElements", ["data", "indices"],
                                ["outputs"],
                                axis=2)
    output = run_node(node_def, [data, indices])
    np.testing.assert_almost_equal(output["outputs"], ref_output)

  def test_gather_nd(self):
    if legacy_opset_pre_ver(11):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support GatherND.".format(
              defs.onnx_opset_version()))

    # valid positive and negative indices for elements
    data = np.array([[0, 1], [2, 3]], dtype=np.int64)
    indices = np.array([[0, 0], [1, 1], [-1, -2]], dtype=np.int64)
    ref_output = np.array([0, 3, 2], dtype=np.int64)
    node_def = helper.make_node("GatherND", ["data", "indices"], ["outputs"])
    output = run_node(node_def, [data, indices])
    np.testing.assert_almost_equal(output["outputs"], ref_output)

    # valid positive and negative indices for slices
    data = np.arange(16, dtype=np.int32).reshape([2, 2, 4])
    indices = np.array([[0, 0], [-1, -2]], dtype=np.int64)
    ref_output = np.array([[0, 1, 2, 3], [8, 9, 10, 11]], dtype=np.int32)
    output = run_node(node_def, [data, indices])
    np.testing.assert_almost_equal(output["outputs"], ref_output)
    indices = np.array([[[0, 0]], [[-1, 0]]], dtype=np.int64)
    ref_output = np.array([[[0, 1, 2, 3]], [[8, 9, 10, 11]]], dtype=np.int32)
    output = run_node(node_def, [data, indices])
    np.testing.assert_almost_equal(output["outputs"], ref_output)

    # indices out of bounds
    indices = np.array([[5, 0], [-1, -3]], dtype=np.int64)
    self.assertRaises(tf.errors.InvalidArgumentError, run_node, node_def,
                      [data, indices])
    indices = np.array([[1, 1, 6], [-2, -1, -9]], dtype=np.int32)
    self.assertRaises(tf.errors.InvalidArgumentError, run_node, node_def,
                      [data, indices])

    if not legacy_opset_pre_ver(12):
      # set batch_dims
      data = np.reshape(np.arange(0, 120, dtype=np.float64), [2, 3, 4, 5])
      indices = np.array(
          [[[1, 2], [0, 1], [-1, 4]], [[-3, -4], [0, -2], [2, 3]]],
          dtype=np.int64)
      ref_output = np.array([[7, 21, 59], [66, 83, 113]], dtype=np.float64)
      node_def = helper.make_node("GatherND", ["data", "indices"], ["outputs"],
                                  batch_dims=2)
      output = run_node(node_def, [data, indices])
      np.testing.assert_almost_equal(output["outputs"], ref_output)

      # indices out of bounds
      indices = np.array(
          [[[4, 1], [0, 1], [-1, 4]], [[-3, -4], [0, -2], [2, 3]]],
          dtype=np.int64)
      self.assertRaises(tf.errors.InvalidArgumentError, run_node, node_def,
                        [data, indices])
      indices = np.array(
          [[[3, 5], [0, 1], [-1, 4]], [[-3, -4], [0, -2], [2, 3]]],
          dtype=np.int64)
      self.assertRaises(tf.errors.InvalidArgumentError, run_node, node_def,
                        [data, indices])

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

    # sys_config.auto_cast=False and x,y,z dtype=uint64 should throw exception
    self.assertRaises(
        RuntimeError, run_node, node_def,
        [x.astype(np.uint64),
         y.astype(np.uint64),
         z.astype(np.uint64)])

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

  def test_greater_or_equal(self):
    if legacy_opset_pre_ver(12):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support GreaterOrEqual.".format(
              defs.onnx_opset_version()))
    node_def = helper.make_node('GreaterOrEqual', ['X', 'Y'], ['Z'])
    shape = [2, 3, 4, 5]
    x = self._get_rnd_int(
        np.iinfo(np.uint8).min,
        np.iinfo(np.uint8).max, shape, np.uint8)
    y = self._get_rnd_int(
        np.iinfo(np.uint8).min,
        np.iinfo(np.uint8).max, shape, np.uint8)
    output = run_node(node_def, [x, y])
    np.testing.assert_equal(output['Z'], np.greater_equal(x, y))
    # test with broadcast
    shape2 = [5]
    x = self._get_rnd_float32(
        np.finfo(np.float16).min,
        np.finfo(np.float16).max, shape).astype(np.float16)
    y = self._get_rnd_float32(
        np.finfo(np.float16).min,
        np.finfo(np.float16).max, shape2).astype(np.float16)
    output = run_node(node_def, [x, y])
    np.testing.assert_equal(output['Z'], np.greater_equal(x, y))

  def test_hardmax(self):
    shape = [2, 3, 4, 5]
    x = self._get_rnd_float32(shape=shape)
    for axis in range(-len(shape), len(shape)):
      node_def = helper.make_node("Hardmax", ["X"], ["Y"], axis=axis)
      output = run_node(node_def, [x])

      axis = axis if axis >= 0 else len(np.shape(x)) + axis
      if axis == len(np.shape(x)) - 1:
        np.testing.assert_almost_equal(output["Y"], tfa.seq2seq.hardmax(x))
      else:
        if not legacy_opset_pre_ver(13):
          y = hardmax.hardmax(x, axis)
          np.testing.assert_almost_equal(output["Y"], y)
        else:
          shape_in_2d = (np.prod(shape[0:axis]).astype(int),
                         np.prod(shape[axis:len(shape)]))
          x_in_2d = np.reshape(x, shape_in_2d)
          y = np.eye(x_in_2d.shape[1], dtype=x.dtype)[np.argmax(x_in_2d,
                                                                axis=1)]
          np.testing.assert_almost_equal(output["Y"], np.reshape(y, shape))

  def test_if(self):
    true_val = helper.make_tensor(name='true_tensor',
                                  data_type=TensorProto.INT64,
                                  dims=(),
                                  vals=[np.int64(1)])
    false_val = helper.make_tensor(name='false_tensor',
                                   data_type=TensorProto.INT64,
                                   dims=(),
                                   vals=[np.int64(0)])
    true_node = helper.make_node('Constant',
                                 inputs=[],
                                 outputs=['true'],
                                 value=true_val)
    false_node = helper.make_node('Constant',
                                  inputs=[],
                                  outputs=['false'],
                                  value=false_val)

    true_out = helper.make_tensor_value_info('true', TensorProto.INT64, [])
    false_out = helper.make_tensor_value_info('false', TensorProto.INT64, [])

    true_graph = helper.make_graph(nodes=[true_node],
                                   name="true_graph",
                                   inputs=[],
                                   outputs=[true_out])
    false_graph = helper.make_graph(nodes=[false_node],
                                    name="false_graph",
                                    inputs=[],
                                    outputs=[false_out])

    node_def = helper.make_node('If', ['cond'], ['outputs'],
                                then_branch=true_graph,
                                else_branch=false_graph)

    for cond, exp in [[True, np.int64(1)], [False, np.int64(0)]]:
      output = run_node(node_def, [cond])
      np.testing.assert_equal(output['outputs'], exp)

    x = self._get_rnd_int(low=-50, high=50, dtype=np.int64)
    y = self._get_rnd_int(low=-50, high=50, dtype=np.int64)
    z = self._get_rnd_int(low=-50, high=50, dtype=np.int64)
    x_val = helper.make_tensor(name='x_tensor',
                               data_type=TensorProto.INT64,
                               dims=(),
                               vals=[x])
    y_val = helper.make_tensor(name='y_tensor',
                               data_type=TensorProto.INT64,
                               dims=(),
                               vals=[y])
    z_val = helper.make_tensor(name='z_tensor',
                               data_type=TensorProto.INT64,
                               dims=(),
                               vals=[z])
    x_node = helper.make_node('Constant', inputs=[], outputs=['x'], value=x_val)
    y_node = helper.make_node('Constant', inputs=[], outputs=['y'], value=y_val)
    z_node = helper.make_node('Constant', inputs=[], outputs=['z'], value=z_val)
    add_node = helper.make_node('Add', inputs=['x', 'y'], outputs=['sum'])
    sub_node = helper.make_node('Sub', inputs=['x', 'y'], outputs=['diff'])
    mul1_node = helper.make_node('Mul', inputs=['sum', 'z'], outputs=['prod1'])
    mul2_node = helper.make_node('Mul', inputs=['diff', 'z'], outputs=['prod2'])

    x_out = helper.make_tensor_value_info('x', TensorProto.INT64, [])
    y_out = helper.make_tensor_value_info('y', TensorProto.INT64, [])
    z_out = helper.make_tensor_value_info('z', TensorProto.INT64, [])
    sum_out = helper.make_tensor_value_info('sum', TensorProto.INT64, [])
    diff_out = helper.make_tensor_value_info('diff', TensorProto.INT64, [])
    prod1_out = helper.make_tensor_value_info('prod1', TensorProto.INT64, [])
    prod2_out = helper.make_tensor_value_info('prod2', TensorProto.INT64, [])

    true_graph = helper.make_graph(nodes=[add_node, mul1_node],
                                   name="true_graph",
                                   inputs=[x_out, y_out, z_out],
                                   outputs=[sum_out, prod1_out])
    false_graph = helper.make_graph(nodes=[sub_node, mul2_node],
                                    name="false_graph",
                                    inputs=[x_out, y_out, z_out],
                                    outputs=[diff_out, prod2_out])

    less_node = helper.make_node('Less', inputs=['x', 'y'], outputs=['cond'])
    if_node = helper.make_node('If',
                               inputs=['cond'],
                               outputs=['result1', 'result2'],
                               then_branch=true_graph,
                               else_branch=false_graph)

    result1_out = helper.make_tensor_value_info('result1', TensorProto.INT64,
                                                [])
    result2_out = helper.make_tensor_value_info('result2', TensorProto.INT64,
                                                [])

    graph = helper.make_graph(
        nodes=[x_node, y_node, z_node, less_node, if_node],
        name="test_if",
        inputs=[],
        outputs=[result1_out, result2_out])

    tf_rep = onnx_graph_to_tensorflow_rep(graph)
    output = tf_rep.run({})
    expected = [x + y, (x + y) * z] if x < y else [x - y, (x - y) * z]
    np.testing.assert_equal(output['result1'], expected[0])
    np.testing.assert_equal(output['result2'], expected[1])

  def test_image_scaler(self):
    if not legacy_opset_pre_ver(9):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support ImageScaler.".format(
              defs.onnx_opset_version()))
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

  def test_greater(self):
    node_def = helper.make_node("Greater", ["X", "Y"], ["Z"])
    x = self._get_rnd_float32(shape=[5, 3, 3, 2])
    y = self._get_rnd_float32(shape=[3, 3, 1])
    output = run_node(node_def, [x, y])
    np.testing.assert_equal(output["Z"],
                            np.greater(x, np.reshape(y, [1, 3, 3, 1])))

  def test_less(self):
    node_def = helper.make_node("Less", ["X", "Y"], ["Z"])
    x = self._get_rnd_float32(shape=[5, 3, 3, 2])
    y = self._get_rnd_float32(shape=[3, 3, 1])
    output = run_node(node_def, [x, y])
    np.testing.assert_equal(output["Z"], np.less(x, np.reshape(y,
                                                               [1, 3, 3, 1])))

  def test_less_or_equal(self):
    if legacy_opset_pre_ver(12):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support LessOrEqual.".format(
              defs.onnx_opset_version()))
    node_def = helper.make_node('LessOrEqual', ['X', 'Y'], ['Z'])
    shape = [2, 3, 4, 5]
    x = self._get_rnd_int(
        np.iinfo(np.int64).min,
        np.iinfo(np.int64).max, shape, np.int64)
    y = self._get_rnd_int(
        np.iinfo(np.int64).min,
        np.iinfo(np.int64).max, shape, np.int64)
    output = run_node(node_def, [x, y])
    np.testing.assert_equal(output['Z'], np.less_equal(x, y))
    # test with broadcast
    shape2 = [5]
    x = self._get_rnd_float32(
        np.finfo(np.float16).min,
        np.finfo(np.float16).max, shape).astype(np.float16)
    y = self._get_rnd_float32(
        np.finfo(np.float16).min,
        np.finfo(np.float16).max, shape2).astype(np.float16)
    output = run_node(node_def, [x, y])
    np.testing.assert_equal(output['Z'], np.less_equal(x, y))
    # test data types that are not natively supported by Tensorflow
    x = self._get_rnd_int(
        np.iinfo(np.uint32).min,
        np.iinfo(np.uint32).max, shape, np.uint32)
    y = self._get_rnd_int(
        np.iinfo(np.uint32).min,
        np.iinfo(np.uint32).max, shape, np.uint32)
    output = run_node(node_def, [x, y])
    np.testing.assert_equal(output['Z'], np.less_equal(x, y))
    x = self._get_rnd_int(
        np.iinfo(np.uint64).min,
        np.iinfo(np.uint64).max, shape, np.uint64)
    y = self._get_rnd_int(
        np.iinfo(np.uint64).min,
        np.iinfo(np.uint64).max, shape, np.uint64)
    self.assertRaises(RuntimeError, run_node, node_def, [x, y])

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
    for device in self._get_device_list():
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
      output = run_node(node_def, [x], device=device)
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

  def test_loop(self):
    # here is the loop testcase
    # while x < y:
    #   x = x + x (v_1)
    #   y = y - x (v_2)
    #   z = x * y (scan_output)
    add_node = helper.make_node('Add', inputs=['x', 'x'], outputs=['sum'])
    sub_node = helper.make_node('Sub', inputs=['y', 'sum'], outputs=['diff'])
    mul_node = helper.make_node('Mul', inputs=['sum', 'diff'], outputs=['prod'])
    less_node = helper.make_node('Less',
                                 inputs=['sum', 'diff'],
                                 outputs=['new_cond'])
    greater_node = helper.make_node('Greater',
                                    inputs=['sum', 'diff'],
                                    outputs=['new_cond'])

    iter_count_in = helper.make_tensor_value_info('iter_count',
                                                  TensorProto.INT64, [])
    cond_in = helper.make_tensor_value_info('cond', TensorProto.BOOL, [])
    cond_int_in = helper.make_tensor_value_info('cond', TensorProto.INT32, [])
    x_in = helper.make_tensor_value_info('x', TensorProto.INT32, [None])
    y_in = helper.make_tensor_value_info('y', TensorProto.INT32, [None])

    cond_out = helper.make_tensor_value_info('cond', TensorProto.BOOL, [])
    new_cond_out = helper.make_tensor_value_info('new_cond', TensorProto.BOOL,
                                                 [])
    sum_out = helper.make_tensor_value_info('sum', TensorProto.INT32, [None])
    diff_out = helper.make_tensor_value_info('diff', TensorProto.INT32, [None])
    prod_out = helper.make_tensor_value_info('prod', TensorProto.INT32, [None])

    v1_initial = np.array([1, 1], dtype=np.int32)
    v2_initial = np.array([100, 100], dtype=np.int32)

    # test for loop
    M = np.array(5, dtype=np.int64)
    cond = np.array(
        True, dtype=np.bool
    )  # value will be ignore because optional "cond" input will be skip
    graph = helper.make_graph(nodes=[add_node, sub_node, mul_node],
                              name="for_loop_graph",
                              inputs=[iter_count_in, cond_in, x_in, y_in],
                              outputs=[cond_out, sum_out, diff_out, prod_out])
    node_def = helper.make_node('Loop', ['M', '', 'v1_initial', 'v2_initial'],
                                ['v1_final', 'v2_final', 'scan_output'],
                                body=graph)
    output = run_node(node_def, [M, cond, v1_initial, v2_initial])
    v1_final = np.array([32, 32], dtype=np.int32)
    v2_final = np.array([38, 38], dtype=np.int32)
    scan_output = np.array(
        [[196, 196], [376, 376], [688, 688], [1120, 1120], [1216, 1216]],
        dtype=np.int32)
    np.testing.assert_almost_equal(output['v1_final'], v1_final)
    np.testing.assert_almost_equal(output['v2_final'], v2_final)
    np.testing.assert_almost_equal(output['scan_output'], scan_output)

    # test while loop
    M = np.array(
        0, dtype=np.int64
    )  # value will be ignore because optional "M" input will be skip
    cond = np.array(np.all(v1_initial < v2_initial), dtype=np.bool)
    graph = helper.make_graph(
        nodes=[add_node, sub_node, mul_node, less_node],
        name="while_loop_graph",
        inputs=[iter_count_in, cond_in, x_in, y_in],
        outputs=[new_cond_out, sum_out, diff_out, prod_out])
    node_def = helper.make_node('Loop',
                                ['', 'cond', 'v1_initial', 'v2_initial'],
                                ['v1_final', 'v2_final', 'scan_output'],
                                body=graph)
    output = run_node(node_def, [M, cond, v1_initial, v2_initial])
    v1_final = np.array([64, 64], dtype=np.int32)
    v2_final = np.array([-26, -26], dtype=np.int32)
    scan_output = np.array([[196, 196], [376, 376], [688, 688], [1120, 1120],
                            [1216, 1216], [-1664, -1664]],
                           dtype=np.int32)
    np.testing.assert_almost_equal(output['v1_final'], v1_final)
    np.testing.assert_almost_equal(output['v2_final'], v2_final)
    np.testing.assert_almost_equal(output['scan_output'], scan_output)

    # test do-while loop
    M = np.array(
        0, dtype=np.int64
    )  # value will be ignore because optional "M" input will be skip
    cond = np.array(1, dtype=np.int32)
    graph = helper.make_graph(
        nodes=[add_node, sub_node, mul_node, greater_node],
        name="do_while_loop_graph",
        inputs=[iter_count_in, cond_int_in, x_in, y_in],
        outputs=[new_cond_out, sum_out, diff_out, prod_out])
    node_def = helper.make_node('Loop',
                                ['', 'cond', 'v1_initial', 'v2_initial'],
                                ['v1_final', 'v2_final', 'scan_output'],
                                body=graph)
    output = run_node(node_def, [M, cond, v1_initial, v2_initial])
    v1_final = np.array([2, 2], dtype=np.int32)
    v2_final = np.array([98, 98], dtype=np.int32)
    scan_output = np.array([[196, 196]], dtype=np.int32)
    np.testing.assert_almost_equal(output['v1_final'], v1_final)
    np.testing.assert_almost_equal(output['v2_final'], v2_final)
    np.testing.assert_almost_equal(output['scan_output'], scan_output)

    # test for loop and while loop conbine
    M = np.array(4, dtype=np.int64)
    cond = np.array(np.all(v1_initial < v2_initial), dtype=np.bool)
    graph = helper.make_graph(
        nodes=[add_node, sub_node, mul_node, less_node],
        name="for_and_while_loop_graph",
        inputs=[iter_count_in, cond_in, x_in, y_in],
        outputs=[new_cond_out, sum_out, diff_out, prod_out])
    node_def = helper.make_node('Loop',
                                ['M', 'cond', 'v1_initial', 'v2_initial'],
                                ['v1_final', 'v2_final', 'scan_output'],
                                body=graph)
    output = run_node(node_def, [M, cond, v1_initial, v2_initial])
    v1_final = np.array([16, 16], dtype=np.int32)
    v2_final = np.array([70, 70], dtype=np.int32)
    scan_output = np.array([[196, 196], [376, 376], [688, 688], [1120, 1120]],
                           dtype=np.int32)
    np.testing.assert_almost_equal(output['v1_final'], v1_final)
    np.testing.assert_almost_equal(output['v2_final'], v2_final)
    np.testing.assert_almost_equal(output['scan_output'], scan_output)

    # test for loop that doesn't run at all (M = 0)
    M = np.array(0, dtype=np.int64)
    cond = np.array(
        True, dtype=np.bool
    )  # value will be ignore because optional "cond" input will be skip
    graph = helper.make_graph(nodes=[add_node, sub_node, mul_node],
                              name="for_loop_graph",
                              inputs=[iter_count_in, cond_in, x_in, y_in],
                              outputs=[cond_out, sum_out, diff_out, prod_out])
    node_def = helper.make_node('Loop', ['M', '', 'v1_initial', 'v2_initial'],
                                ['v1_final', 'v2_final', 'scan_output'],
                                body=graph)
    output = run_node(node_def, [M, cond, v1_initial, v2_initial])
    v1_final = np.array([1, 1], dtype=np.int32)
    v2_final = np.array([100, 100], dtype=np.int32)
    scan_output = np.array([], dtype=np.int32).reshape([0, 0])
    np.testing.assert_almost_equal(output['v1_final'], v1_final)
    np.testing.assert_almost_equal(output['v2_final'], v2_final)
    np.testing.assert_equal(output['scan_output'], scan_output)

    # test while loop that doesn't run at all (cond = False)
    M = np.array(
        0, dtype=np.int64
    )  # value will be ignore because optional "M" input will be skip
    cond = np.array(False, dtype=np.bool)
    graph = helper.make_graph(
        nodes=[add_node, sub_node, mul_node, less_node],
        name="while_loop_graph",
        inputs=[iter_count_in, cond_in, x_in, y_in],
        outputs=[new_cond_out, sum_out, diff_out, prod_out])
    node_def = helper.make_node('Loop',
                                ['', 'cond', 'v1_initial', 'v2_initial'],
                                ['v1_final', 'v2_final', 'scan_output'],
                                body=graph)
    output = run_node(node_def, [M, cond, v1_initial, v2_initial])
    v1_final = np.array([1, 1], dtype=np.int32)
    v2_final = np.array([100, 100], dtype=np.int32)
    scan_output = np.array([], dtype=np.int32).reshape([0, 0])
    np.testing.assert_almost_equal(output['v1_final'], v1_final)
    np.testing.assert_almost_equal(output['v2_final'], v2_final)
    np.testing.assert_almost_equal(output['scan_output'], scan_output)

    # test while loop that doesn't have any scan_outputs
    M = np.array(4, dtype=np.int64)
    cond = np.array(np.all(v1_initial < v2_initial), dtype=np.bool)
    graph = helper.make_graph(nodes=[add_node, sub_node, mul_node, less_node],
                              name="while_loop_graph",
                              inputs=[iter_count_in, cond_in, x_in, y_in],
                              outputs=[new_cond_out, sum_out, diff_out])
    node_def = helper.make_node('Loop',
                                ['M', 'cond', 'v1_initial', 'v2_initial'],
                                ['v1_final', 'v2_final'],
                                body=graph)
    output = run_node(node_def, [M, cond, v1_initial, v2_initial])
    v1_final = np.array([16, 16], dtype=np.int32)
    v2_final = np.array([70, 70], dtype=np.int32)
    np.testing.assert_almost_equal(output['v1_final'], v1_final)
    np.testing.assert_almost_equal(output['v2_final'], v2_final)

    # test for loop that doesn't run at all (M = 0)
    # and the scan_outputs shape is not the same as the inputs
    v1_initial = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.int32)
    v3_initial = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.int32)
    matmul_node = helper.make_node('MatMul',
                                   inputs=['x', 'z'],
                                   outputs=['product'])
    x_in = helper.make_tensor_value_info('x', TensorProto.INT32, [None, None])
    z_in = helper.make_tensor_value_info('z', TensorProto.INT32, [None, None])
    sum_out = helper.make_tensor_value_info('sum', TensorProto.INT32,
                                            [None, None])
    z_out = helper.make_tensor_value_info('z', TensorProto.INT32, [None, None])
    product_out = helper.make_tensor_value_info('product', TensorProto.INT32,
                                                [None, None])

    M = np.array(0, dtype=np.int64)
    cond = np.array(
        True, dtype=np.bool
    )  # value will be ignore because optional "cond" input will be skip
    graph = helper.make_graph(nodes=[add_node, matmul_node],
                              name="for_loop_graph",
                              inputs=[iter_count_in, cond_in, x_in, z_in],
                              outputs=[cond_out, sum_out, z_out, product_out])
    node_def = helper.make_node('Loop', ['M', '', 'v1_initial', 'v3_initial'],
                                ['v1_final', 'v3_final', 'scan_output'],
                                body=graph)
    scan_output = np.array([], dtype=np.int32).reshape([0, 0, 0])
    output = run_node(node_def, [M, cond, v1_initial, v3_initial])
    np.testing.assert_almost_equal(output['v1_final'], v1_initial)
    np.testing.assert_almost_equal(output['v3_final'], v3_initial)
    np.testing.assert_almost_equal(output['scan_output'], scan_output)

    # verify infinite loop will get exception
    M = np.array(
        0, dtype=np.int64
    )  # value will be ignore because optional "M" input will be skip
    cond = np.array(
        True, dtype=np.bool
    )  # value will be ignore because optional "cond" input will be skip
    graph = helper.make_graph(nodes=[add_node, sub_node, mul_node, less_node],
                              name="while_loop_graph",
                              inputs=[iter_count_in, cond_in, x_in, y_in],
                              outputs=[cond_out, sum_out, diff_out, prod_out])
    node_def = helper.make_node('Loop', ['', '', 'v1_initial', 'v2_initial'],
                                ['v1_final', 'v2_final', 'scan_output'],
                                body=graph)
    try:
      run_node(node_def, [M, cond, v1_initial, v2_initial])
      raise AssertionError("Expected RuntimeError not raise when Loop inputs " +
                           "M and cond are both not set at the same time")
    except RuntimeError as e:
      assert "M and cond in Loop are not set" in str(e)

  def test_matmul(self):
    node_def = helper.make_node("MatMul", ["A", "B"], ["Y"])
    a = self._get_rnd_float32(shape=[5, 6])
    b = self._get_rnd_float32(shape=[6, 5])
    output = run_node(node_def, [a, b])
    np.testing.assert_allclose(output["Y"],
                               np.matmul(a, b),
                               rtol=1e-6,
                               atol=1e-6)
    # test data types that are not natively supported by Tensorflow
    a = self._get_rnd_int(0, 1000, [10, 10], np.uint32)
    b = self._get_rnd_int(0, 1000, [10, 10], np.uint32)
    output = run_node(node_def, [a, b])
    np.testing.assert_allclose(output["Y"],
                               np.matmul(a, b),
                               rtol=1e-6,
                               atol=1e-6)
    # sys_config.auto_cast=False and a or b dtype=uint64 should throw exception
    self.assertRaises(
        RuntimeError, run_node, node_def,
        [a.astype(np.uint64), b.astype(np.uint64)])

  def test_matmul_integer(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support MatMulInteger.".format(
              defs.onnx_opset_version()))

    node_def = helper.make_node("MatMulInteger",
                                ["A", "B", "a_zero_point", "b_zero_point"],
                                ["Z"])
    lower_bound = {np.uint8: 0, np.int8: -20}
    for dtype in [np.uint8, np.int8]:
      # A & B are 3-D tensor and a_zero_point & b_zero_point are scalar
      A = self._get_rnd_int(lower_bound[dtype],
                            20,
                            shape=(2, 3, 4),
                            dtype=dtype)
      B = self._get_rnd_int(lower_bound[dtype],
                            20,
                            shape=(2, 4, 6),
                            dtype=dtype)
      a_zero_point = self._get_rnd_int(lower_bound[dtype], 20, dtype=dtype)
      b_zero_point = self._get_rnd_int(lower_bound[dtype], 20, dtype=dtype)
      A_minus_zero_point = np.subtract(A.astype(np.int32),
                                       a_zero_point.astype(np.int32))
      B_minus_zero_point = np.subtract(B.astype(np.int32),
                                       b_zero_point.astype(np.int32))
      z = np.matmul(A_minus_zero_point, B_minus_zero_point)
      output = run_node(node_def, [A, B, a_zero_point, b_zero_point])
      np.testing.assert_almost_equal(output["Z"], z)
      # A & B are 4-D tensor and a_zero_point & b_zero_point are 1-D tensor
      A = self._get_rnd_int(lower_bound[dtype],
                            20,
                            shape=(2, 5, 3, 4),
                            dtype=dtype)
      B = self._get_rnd_int(lower_bound[dtype],
                            20,
                            shape=(2, 1, 4, 6),
                            dtype=dtype)
      a_zero_point = self._get_rnd_int(lower_bound[dtype],
                                       20,
                                       shape=(A.shape[-2]),
                                       dtype=dtype)
      b_zero_point = self._get_rnd_int(lower_bound[dtype],
                                       20,
                                       shape=(B.shape[-1]),
                                       dtype=dtype)
      a_zero_point_with_reshape = np.reshape(a_zero_point, [A.shape[-2], 1])
      A_minus_zero_point = np.subtract(
          A.astype(np.int32), a_zero_point_with_reshape.astype(np.int32))
      B_minus_zero_point = np.subtract(B.astype(np.int32),
                                       b_zero_point.astype(np.int32))
      z = np.matmul(A_minus_zero_point, B_minus_zero_point)
      output = run_node(node_def, [A, B, a_zero_point, b_zero_point])
      np.testing.assert_almost_equal(output["Z"], z)

    node_def = helper.make_node("MatMulInteger", ["A", "B"], ["Z"])
    for dtype in [np.uint8, np.int8]:
      # A & B are 3-D tensor
      A = self._get_rnd_int(lower_bound[dtype],
                            20,
                            shape=(2, 3, 4),
                            dtype=dtype)
      B = self._get_rnd_int(lower_bound[dtype],
                            20,
                            shape=(2, 4, 6),
                            dtype=dtype)
      z = np.matmul(A.astype(np.int32), B.astype(np.int32))
      output = run_node(node_def, [A, B])
      np.testing.assert_almost_equal(output["Z"], z)
      # A & B are 4-D tensor
      A = self._get_rnd_int(lower_bound[dtype],
                            20,
                            shape=(2, 5, 3, 4),
                            dtype=dtype)
      B = self._get_rnd_int(lower_bound[dtype],
                            20,
                            shape=(2, 1, 4, 6),
                            dtype=dtype)
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

  def _test_pooling(self,
                    input_shape,
                    kernel_shape,
                    strides=None,
                    dilations=None,
                    pads=None,
                    auto_pad=None,
                    ceil_mode=None,
                    count_include_pad=None,
                    pooling_type="MAX",
                    input_dtype=np.float32,
                    p=None):

    for device in self._get_device_list():
      op = "MaxPool" if pooling_type.upper().startswith("MAX") else \
           "AveragePool" if pooling_type.upper() == "AVG" else "LpPool"
      node_def_kwargs = {
          "op_type": op,
          "inputs": ["X"],
          "outputs": ["Y"],
          "kernel_shape": kernel_shape
      }

      if strides is not None:
        node_def_kwargs["strides"] = strides
      if dilations is not None:
        node_def_kwargs["dilations"] = dilations
      if pads is not None:
        node_def_kwargs["pads"] = pads
      orig_pads = pads  # save it for the 2nd loop
      if auto_pad is not None:
        node_def_kwargs["auto_pad"] = auto_pad
        pads = auto_pad
      orig_ceil_mode = ceil_mode  # save it for the 2nd loop
      if ceil_mode is not None:
        node_def_kwargs["ceil_mode"] = ceil_mode
      else:
        ceil_mode = 0
      if count_include_pad is not None:
        node_def_kwargs["count_include_pad"] = count_include_pad
      if p is not None:
        node_def_kwargs["p"] = p

      node_def = helper.make_node(**node_def_kwargs)

      if input_dtype == np.float32:
        x = self._get_rnd_float32(shape=input_shape)
      else:
        x = self._get_rnd_int(low=np.iinfo(input_dtype).min,
                              high=np.iinfo(input_dtype).max,
                              shape=input_shape,
                              dtype=input_dtype)

      output = run_node(node_def, [x], device=device)

      test_output = py_pool(x,
                            kernel_shape=kernel_shape,
                            strides=strides,
                            dilations=dilations,
                            padding=pads,
                            ceil_mode=ceil_mode,
                            pooling_type=pooling_type,
                            include_indices=False,
                            p=p)

      np.testing.assert_almost_equal(output["Y"],
                                     test_output,
                                     decimal=5 if pooling_type == "LP" else 7)

      # set pads and ceil_mode values back to the original values for the 2nd loop
      pads = orig_pads
      ceil_mode = orig_ceil_mode

  def test_max_pool_2d(self):
    kernel_shape = [1, 2]
    strides = [1, 2]

    input_shape = [10, 10, 4, 4]
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides)

  def test_max_pool_2d_same_lower(self):
    kernel_shape = [1, 2]
    strides = [1, 2]
    auto_pad = "SAME_LOWER"

    input_shape = [10, 10, 7, 7]
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       auto_pad=auto_pad)

  def test_max_pool_2d_ceil_same_lower(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support ceil mode.".format(
              defs.onnx_opset_version()))

    kernel_shape = [2, 1]
    strides = [1, 2]
    auto_pad = "SAME_LOWER"
    ceil_mode = 1

    input_shape = [10, 10, 7, 7]
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       auto_pad=auto_pad,
                       ceil_mode=ceil_mode)

  def test_max_pool_2d_same_upper(self):
    kernel_shape = [1, 2]
    strides = [1, 2]
    auto_pad = "SAME_UPPER"

    input_shape = [10, 10, 7, 7]
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       auto_pad=auto_pad)

  def test_max_pool_2d_ceil(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support ceil mode.".format(
              defs.onnx_opset_version()))

    kernel_shape = [3, 3]
    strides = [2, 2]
    ceil_mode = 1

    input_shape = [10, 3, 24, 24]
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       ceil_mode=ceil_mode)

  def test_max_pool_2d_dilations(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support dilations.".format(
              defs.onnx_opset_version()))

    kernel_shape = [3, 3]
    strides = [2, 2]
    dilations = [3, 3]

    input_shape = [10, 3, 24, 24]
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       dilations=dilations)

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
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       dilations=dilations,
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
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       dilations=dilations,
                       pads=pads)

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
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       dilations=dilations,
                       pads=pads,
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
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       dilations=dilations,
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
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       dilations=dilations,
                       auto_pad=auto_pad)

  def test_max_pool_2d_dilations_ceil_pads_int8(self):
    if legacy_opset_pre_ver(12):
      raise unittest.SkipTest(
          "ONNX version {} does not support int8 input type.".format(
              defs.onnx_opset_version()))

    kernel_shape = [3, 3]
    strides = [2, 2]
    dilations = [3, 3]
    pads = [1, 1, 2, 2]
    ceil_mode = 1

    input_shape = [10, 3, 23, 23]
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       dilations=dilations,
                       pads=pads,
                       ceil_mode=ceil_mode,
                       input_dtype=np.int8)

  def test_max_pool_3d(self):
    kernel_shape = [3, 3, 3]
    strides = [2, 2, 2]

    input_shape = [10, 3, 23, 23, 23]
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
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
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       dilations=dilations,
                       pads=pads,
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
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       dilations=dilations,
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
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       dilations=dilations,
                       pads=pads,
                       ceil_mode=ceil_mode)

  def test_max_pool_1d(self):
    kernel_shape = [3]
    strides = [2]

    input_shape = [10, 3, 23]
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides)

  def test_max_pool_with_argmax_2d_dilations_ceil_pads(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support dilations nor ceil mode.".format(
              defs.onnx_opset_version()))
    for device in self._get_device_list():
      kernel_shape = [3, 3]
      strides = [2, 2]
      dilations = [3, 3]
      pads = [1, 1, 2, 2]
      ceil_mode = True
      node_def = helper.make_node("MaxPool", ["X"], ["Y", "Ind"],
                                  kernel_shape=kernel_shape,
                                  strides=strides,
                                  dilations=dilations,
                                  pads=pads,
                                  ceil_mode=ceil_mode)

      input_shape = [10, 3, 23, 23]
      x = self._get_rnd_float32(shape=input_shape) - 2

      output = run_node(node_def, [x], device=device)

      test_output, test_ind = py_pool(x,
                                      kernel_shape=kernel_shape,
                                      strides=strides,
                                      dilations=dilations,
                                      padding=pads,
                                      ceil_mode=ceil_mode,
                                      pooling_type="MAX")

      np.testing.assert_almost_equal(output["Y"], test_output)
      np.testing.assert_almost_equal(output["Ind"], test_ind)

  def test_max_pool_with_argmax_3d(self):
    kernel_shape = [3, 3, 3]
    strides = [2, 2, 2]
    node_def = helper.make_node("MaxPool", ["X"], ["Y", "Ind"],
                                kernel_shape=kernel_shape,
                                strides=strides)

    input_shape = [10, 1, 23, 23, 23]
    x = self._get_rnd_float32(shape=input_shape)
    self.assertRaises(RuntimeError, run_node, node_def, [x])

  def test_max_pool_4d(self):
    kernel_shape = [3, 3, 3, 3]
    strides = [2, 2, 2, 2]
    node_def = helper.make_node("MaxPool", ["X"], ["Y", "Ind"],
                                kernel_shape=kernel_shape,
                                strides=strides)

    input_shape = [1, 1, 4, 4, 4, 4]
    x = self._get_rnd_float32(shape=input_shape)
    self.assertRaises(RuntimeError, run_node, node_def, [x])

  def test_max_unpool(self):
    for device in self._get_device_list():
      input_shape = [10, 10, 4, 4]
      x = self._get_rnd_float32(shape=input_shape)

      node_def = helper.make_node("MaxPool", ["X"], ["Pool", "Indices"],
                                  kernel_shape=[2, 2],
                                  strides=[2, 2])
      output_pool = run_node(node_def, [x], device=device)

      node_def = helper.make_node("MaxUnpool", ["Pool", "Indices"], ["Y"],
                                  kernel_shape=[2, 2],
                                  strides=[2, 2])
      output_unpool = run_node(node_def,
                               [output_pool["Pool"], output_pool["Indices"]],
                               device=device)

      test_output = np.zeros(input_shape)
      for i1 in range(0, input_shape[0]):
        for i2 in range(0, input_shape[1]):
          for i3 in range(0, input_shape[2], 2):
            for i4 in range(0, input_shape[3], 2):
              max_val = float('-inf')
              for j1 in range(i3, i3 + 2):
                for j2 in range(i4, i4 + 2):
                  if x[i1][i2][j1][j2] > max_val:
                    max_val = x[i1][i2][j1][j2]
                    max_ind = (j1, j2)
              j1, j2 = max_ind
              test_output[i1][i2][j1][j2] = max_val
      np.testing.assert_almost_equal(output_unpool["Y"], test_output)

  def test_average_pool_1d(self):
    kernel_shape = [3]
    strides = [2]

    input_shape = [10, 3, 23]
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       pooling_type="AVG")

  def test_average_pool_2d(self):
    kernel_shape = [1, 2]
    strides = [1, 2]

    input_shape = [10, 10, 4, 4]
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       pooling_type="AVG")

  def test_average_pool_2d_same_upper(self):
    kernel_shape = [1, 2]
    strides = [1, 2]
    auto_pad = "SAME_UPPER"

    input_shape = [10, 10, 7, 7]
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       auto_pad=auto_pad,
                       pooling_type="AVG")

  def test_average_pool_3d(self):
    kernel_shape = [3, 3, 3]
    strides = [2, 2, 2]

    input_shape = [10, 3, 23, 23, 23]
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       pooling_type="AVG")

  def test_lp2_pool_2d(self):
    kernel_shape = [1, 2]
    strides = [1, 2]
    p = 2

    input_shape = [10, 10, 4, 4]
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       pooling_type="LP",
                       p=p)

  def test_lp2_pool_2d_same_lower(self):
    kernel_shape = [1, 2]
    strides = [1, 2]
    p = 2
    auto_pad = "SAME_LOWER"

    input_shape = [10, 10, 7, 7]
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       auto_pad=auto_pad,
                       pooling_type="LP",
                       p=p)

  def test_lp2_pool_2d_same_upper(self):
    kernel_shape = [1, 2]
    strides = [1, 2]
    p = 2
    auto_pad = "SAME_UPPER"

    input_shape = [10, 10, 7, 7]
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       auto_pad=auto_pad,
                       pooling_type="LP",
                       p=p)

  def test_lp2_pool_2d_pads(self):
    kernel_shape = [3, 3]
    strides = [2, 2]
    p = 2
    pads = [1, 1, 2, 2]

    input_shape = [10, 3, 24, 24]
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       pads=pads,
                       pooling_type="LP",
                       p=p)

  def test_lp2_pool_3d(self):
    kernel_shape = [3, 3, 3]
    strides = [2, 2, 2]
    p = 2

    input_shape = [10, 3, 23, 23, 23]
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       pooling_type="LP",
                       p=p)

  def test_lp2_pool_1d(self):
    kernel_shape = [3]
    strides = [2]
    p = 2

    input_shape = [10, 3, 23]
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       pooling_type="LP",
                       p=p)

  def test_lp3_pool_2d_pads(self):
    kernel_shape = [3, 3]
    strides = [2, 2]
    p = 3
    pads = [1, 1, 2, 2]

    input_shape = [10, 3, 24, 24]
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       pads=pads,
                       pooling_type="LP",
                       p=p)

  def test_lp3_pool_3d(self):
    kernel_shape = [3, 3, 3]
    strides = [2, 2, 2]
    p = 3

    input_shape = [10, 3, 23, 23, 23]
    self._test_pooling(input_shape=input_shape,
                       kernel_shape=kernel_shape,
                       strides=strides,
                       pooling_type="LP",
                       p=p)

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
    # test data cast for int16
    x = self._get_rnd_int(shape=[5, 5], low=1, high=100, dtype=np.int16)
    y = self._get_rnd_int(shape=[5, 5], low=1, high=100, dtype=np.int16)
    node_def = helper.make_node("Mod", ["X", "Y"], ["Z"], fmod=0)
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.mod(x, y))

  def test_multinomial(self):
    node_def = helper.make_node("Multinomial", ["X"], ["Y"],
                                sample_size=5,
                                dtype=TensorProto.INT64)
    x = np.array([[math.log(0.5), math.log(0.5)]], dtype=np.float32)
    output = run_node(node_def, [x])
    # the output shape should be [1, 5]
    np.testing.assert_equal(output["Y"].shape, [1, 5])
    # the output dtype should be int64
    np.testing.assert_equal(output["Y"].dtype, np.int64)

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

    # with default axis
    indices = np.array([[0, 2], [1, 2], [0, 1]])
    depth = np.int32(5)
    on_value = 6.0
    off_value = 2.0
    values = np.array([off_value, on_value])
    node_def = helper.make_node('OneHot',
                                inputs=['indices', 'depth', 'values'],
                                outputs=['y'])
    y = one_hot(indices, depth, dtype=values.dtype)
    y = y * (on_value - off_value) + off_value
    output = run_node(node_def, inputs=[indices, depth, values])
    np.testing.assert_equal(output['y'], y)
    # test data types that are not natively supported by tensorflow
    output = run_node(
        node_def, inputs=[indices.astype(np.uint16),
                          np.uint16(depth), values])
    np.testing.assert_equal(output['y'], y)
    self.assertRaises(RuntimeError, run_node, node_def,
                      [indices.astype(np.uint64), depth, values])
    self.assertRaises(RuntimeError, run_node, node_def,
                      [indices, np.int64(depth), values])

    # with axis
    axis = 1
    indices = np.array([[0, 9], [3, 7], [5, 2]])
    depth = np.int32(10)
    on_value = 8
    off_value = -1
    values = np.array([off_value, on_value], np.int8)
    node_def = helper.make_node('OneHot',
                                inputs=['indices', 'depth', 'values'],
                                outputs=['y'],
                                axis=axis)
    y = one_hot(indices, depth, axis=axis, dtype=values.dtype)
    y = y * (on_value - off_value) + off_value
    output = run_node(node_def, inputs=[indices, depth, values])
    np.testing.assert_equal(output['y'], y)

    # with negative indices and negative axis
    axis = -3
    indices = np.array([[0, -9], [-3, 7], [5, -2]])
    depth = np.int32(10)
    on_value = 4
    off_value = 1
    values = np.array([off_value, on_value], np.int16)
    node_def = helper.make_node('OneHot',
                                inputs=['indices', 'depth', 'values'],
                                outputs=['y'],
                                axis=axis)
    y = one_hot(indices, depth, axis=axis, dtype=values.dtype)
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

  def test_resize(self):
    if legacy_opset_pre_ver(11):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support Resize with attributes: " +
          "coordinate_transformation_mode, cubic_coeff_a, exclude_outside, " +
          "extrapolation_value, nearest_mode and inputs: roi and sizes".format(
              defs.onnx_opset_version()))
    data = np.reshape(np.arange(1, 101, dtype=np.float32), [1, 1, 10, 10])
    roi = np.array([], dtype=np.float32)

    # resize_nearest_round_prefer_ceil_align_corners_scales
    node_def = helper.make_node("Resize",
                                inputs=['X', 'roi', 'scales'],
                                outputs=['Y'],
                                coordinate_transformation_mode='align_corners',
                                mode='nearest',
                                nearest_mode='round_prefer_ceil')
    scales = np.array([1, 1, 0.9, 0.9], dtype=np.float32)
    expected = np.array(
        [[[[1, 2, 3, 4, 6, 7, 8, 9, 10], [11, 12, 13, 14, 16, 17, 18, 19, 20],
           [21, 22, 23, 24, 26, 27, 28, 29, 30],
           [31, 32, 33, 34, 36, 37, 38, 39, 40],
           [51, 52, 53, 54, 56, 57, 58, 59, 60],
           [61, 62, 63, 64, 66, 67, 68, 69, 70],
           [71, 72, 73, 74, 76, 77, 78, 79, 80],
           [81, 82, 83, 84, 86, 87, 88, 89, 90],
           [91, 92, 93, 94, 96, 97, 98, 99, 100]]]],
        dtype=np.float32)  # expected value is calculated by onnx-runtime
    output = run_node(node_def, [data, roi, scales])
    np.testing.assert_almost_equal(output["Y"], expected)

    # resize_nearest_round_prefer_ceil_align_corners_sizes
    node_def = helper.make_node("Resize",
                                inputs=['X', 'roi', 'scales', 'sizes'],
                                outputs=['Y'],
                                coordinate_transformation_mode='align_corners',
                                mode='nearest',
                                nearest_mode='round_prefer_ceil')
    x = np.reshape(np.arange(1, 151, dtype=np.float32), [2, 3, 5, 5])
    scales = np.array([], dtype=np.float32)
    sizes = np.array([2, 3, 4, 4], dtype=np.int64)
    expected = np.array(
        [[[[1, 2, 4, 5], [6, 7, 9, 10], [16, 17, 19, 20], [21, 22, 24, 25]],
          [[26, 27, 29, 30], [31, 32, 34, 35], [41, 42, 44, 45],
           [46, 47, 49, 50.]],
          [[51, 52, 54, 55], [56, 57, 59, 60], [66, 67, 69, 70],
           [71, 72, 74, 75]]],
         [[[76, 77, 79, 80], [81, 82, 84, 85], [91, 92, 94, 95],
           [96, 97, 99, 100]],
          [[101, 102, 104, 105], [106, 107, 109, 110], [116, 117, 119, 120],
           [121, 122, 124, 125]],
          [[126, 127, 129, 130], [131, 132, 134, 135], [141, 142, 144, 145],
           [146, 147, 149, 150]]]],
        dtype=np.float32)  # expected value is calculated by onnx-runtime
    output = run_node(node_def, [x, roi, scales, sizes])
    np.testing.assert_almost_equal(output["Y"], expected)

    # resize_nearest_floor_asymmetric_scales
    node_def = helper.make_node("Resize",
                                inputs=['X', 'roi', 'scales'],
                                outputs=['Y'],
                                coordinate_transformation_mode='asymmetric',
                                mode='nearest',
                                nearest_mode='floor')
    scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)
    expected = np.array(
        [[[[1, 2, 3, 4, 6, 7, 8, 9], [11, 12, 13, 14, 16, 17, 18, 19],
           [21, 22, 23, 24, 26, 27, 28, 29], [31, 32, 33, 34, 36, 37, 38, 39],
           [51, 52, 53, 54, 56, 57, 58, 59], [61, 62, 63, 64, 66, 67, 68, 69],
           [71, 72, 73, 74, 76, 77, 78, 79], [81, 82, 83, 84, 86, 87, 88, 89]]]
        ],
        dtype=np.float32)  # expected value is calculated by onnx-runtime
    output = run_node(node_def, [data, roi, scales])
    np.testing.assert_almost_equal(output["Y"], expected)

    # resize_nearest_floor_asymmetric_sizes
    node_def = helper.make_node("Resize",
                                inputs=['X', 'roi', 'scales', 'sizes'],
                                outputs=['Y'],
                                coordinate_transformation_mode='asymmetric',
                                mode='nearest',
                                nearest_mode='floor')
    scales = np.array([], dtype=np.float32)
    sizes = np.array([1, 1, 7, 7], dtype=np.int64)
    expected = np.array(
        [[[[1, 2, 3, 5, 6, 8, 9], [11, 12, 13, 15, 16, 18, 19],
           [21, 22, 23, 25, 26, 28, 29], [41, 42, 43, 45, 46, 48, 49],
           [51, 52, 53, 55, 56, 58, 59], [71, 72, 73, 75, 76, 78, 79],
           [81, 82, 83, 85, 86, 88, 89]]]],
        dtype=np.float32)  # expected value is calculated by onnx-runtime
    output = run_node(node_def, [data, roi, scales, sizes])
    np.testing.assert_almost_equal(output["Y"], expected)

    # resize_nearest_floor_half_pixel_scales
    node_def = helper.make_node(
        "Resize",
        inputs=['X', 'roi', 'scales'],
        outputs=['Y'],
        coordinate_transformation_mode='tf_half_pixel_for_nn',
        mode='nearest',
        nearest_mode='floor')
    scales = np.array([1, 1, 0.8, 0.8], dtype=np.float32)
    expected = np.array(
        [[[[1, 2, 4, 5, 6, 7, 9, 10], [11, 12, 14, 15, 16, 17, 19, 20],
           [31, 32, 34, 35, 36, 37, 39, 40], [41, 42, 44, 45, 46, 47, 49, 50],
           [51, 52, 54, 55, 56, 57, 59, 60], [61, 62, 64, 65, 66, 67, 69, 70],
           [81, 82, 84, 85, 86, 87, 89, 90], [91, 92, 94, 95, 96, 97, 99, 100]]]
        ],
        dtype=np.float32)  # expected value is calculated by onnx-runtime
    output = run_node(node_def, [data, roi, scales])
    np.testing.assert_almost_equal(output["Y"], expected)

    # resize_nearest_floor_half_pixel_sizes
    node_def = helper.make_node(
        "Resize",
        inputs=['X', 'roi', 'scales', 'sizes'],
        outputs=['Y'],
        coordinate_transformation_mode='tf_half_pixel_for_nn',
        mode='nearest',
        nearest_mode='floor')
    scales = np.array([], dtype=np.float32)
    sizes = np.array([1, 1, 7, 7], dtype=np.int64)
    expected = np.array(
        [[[[1, 3, 4, 6, 7, 8, 10], [21, 23, 24, 26, 27, 28, 30],
           [31, 33, 34, 36, 37, 38, 40], [51, 53, 54, 56, 57, 58, 60],
           [61, 63, 64, 66, 67, 68, 70], [71, 73, 74, 76, 77, 78, 80],
           [91, 93, 94, 96, 97, 98, 100]]]],
        dtype=np.float32)  # expected value is calculated by onnx-runtime
    output = run_node(node_def, [data, roi, scales, sizes])
    np.testing.assert_almost_equal(output["Y"], expected)

    # resize_linear_align_corners_scales
    node_def = helper.make_node("Resize",
                                inputs=['X', 'roi', 'scales'],
                                outputs=['Y'],
                                coordinate_transformation_mode="align_corners",
                                mode='linear')
    scales = np.array([1, 1, 0.8, 0.8], dtype=np.float32)
    expected = np.array(
        [[[[
            1., 2.2857141, 3.5714285, 4.857143, 6.142857, 7.428571, 8.714286,
            10.
        ],
           [
               13.857142, 15.142857, 16.428572, 17.714287, 19., 20.285715,
               21.571428, 22.857143
           ],
           [
               26.714287, 28., 29.285713, 30.571426, 31.857141, 33.142857,
               34.428574, 35.714283
           ],
           [
               39.57143, 40.857143, 42.14286, 43.428574, 44.714287, 46.,
               47.285717, 48.57143
           ],
           [
               52.428574, 53.714283, 55., 56.285713, 57.571426, 58.857143,
               60.142857, 61.428566
           ],
           [
               65.28571, 66.57143, 67.85714, 69.14285, 70.428566, 71.71428, 73.,
               74.28571
           ],
           [
               78.14286, 79.42857, 80.71428, 82., 83.28572, 84.57143, 85.85715,
               87.14286
           ],
           [
               91., 92.28571, 93.57143, 94.85715, 96.14285, 97.42857, 98.71429,
               100.
           ]]]],
        dtype=np.float32)  # expected value is calculated by onnx-runtime
    output = run_node(node_def, [data, roi, scales])
    np.testing.assert_allclose(output['Y'], expected, rtol=1e-6, atol=1e-6)

    # resize_linear_align_corners_sizes
    node_def = helper.make_node("Resize",
                                inputs=['X', 'roi', 'scales', 'sizes'],
                                outputs=['Y'],
                                coordinate_transformation_mode="align_corners",
                                mode='linear')
    scales = np.array([], dtype=np.float32)
    sizes = np.array([1, 1, 7, 7], dtype=np.int64)
    expected = np.array(
        [[[[1., 2.5, 4., 5.5, 7., 8.5, 10.],
           [16., 17.5, 19., 20.5, 22., 23.5, 25.],
           [31., 32.5, 34., 35.5, 37., 38.5, 40.],
           [46., 47.5, 49., 50.5, 52., 53.5, 55.],
           [61., 62.5, 64., 65.5, 67., 68.5, 70.],
           [76., 77.5, 79., 80.5, 82., 83.5, 85.],
           [91., 92.5, 94., 95.5, 97., 98.5, 100.]]]],
        dtype=np.float32)  # expected value is calculated by onnx-runtime
    output = run_node(node_def, [data, roi, scales, sizes])
    np.testing.assert_almost_equal(output["Y"], expected)

    # resize_linear_asymmetric_scales
    node_def = helper.make_node("Resize",
                                inputs=['X', 'roi', 'scales'],
                                outputs=['Y'],
                                coordinate_transformation_mode="asymmetric",
                                mode='linear')
    scales = np.array([1, 1, 0.8, 0.8], dtype=np.float32)
    expected = np.array(
        [[[[1., 2.25, 3.5, 4.75, 6., 7.25, 8.5, 9.75],
           [13.5, 14.75, 16., 17.25, 18.5, 19.75, 21., 22.25],
           [26., 27.25, 28.5, 29.75, 31., 32.25, 33.5, 34.75],
           [38.5, 39.75, 41., 42.25, 43.5, 44.75, 46., 47.25],
           [51., 52.25, 53.5, 54.75, 56., 57.25, 58.5, 59.75],
           [63.5, 64.75, 66., 67.25, 68.5, 69.75, 71., 72.25],
           [76., 77.25, 78.5, 79.75, 81., 82.25, 83.5, 84.75],
           [88.5, 89.75, 91., 92.25, 93.5, 94.75, 96., 97.25]]]],
        dtype=np.float32)  # expected value is calculated by onnx-runtime
    output = run_node(node_def, [data, roi, scales])
    np.testing.assert_almost_equal(output["Y"], expected)

    # resize_linear_asymmetric_sizes
    node_def = helper.make_node("Resize",
                                inputs=['X', 'roi', 'scales', 'sizes'],
                                outputs=['Y'],
                                coordinate_transformation_mode="asymmetric",
                                mode='linear')
    scales = np.array([], dtype=np.float32)
    sizes = np.array([1, 1, 7, 7], dtype=np.int64)
    expected = np.array([[
        [[1., 2.4285715, 3.857143, 5.285714, 6.714286, 8.142857, 9.571428],
         [15.285715, 16.714287, 18.142857, 19.571428, 21., 22.42857, 23.857141],
         [29.571428, 31., 32.428574, 33.857143, 35.285717, 36.714287, 38.14286],
         [43.857143, 45.28571, 46.714283, 48.14286, 49.571434, 51., 52.42857],
         [
             58.14286, 59.57143, 61.000004, 62.428574, 63.857143, 65.28572,
             66.71429
         ], [72.42857, 73.85713, 75.28572, 76.71429, 78.14286, 79.57143, 81.],
         [86.71429, 88.14285, 89.57143, 91., 92.42857, 93.85714, 95.28571]]
    ]],
                        dtype=np.float32
                       )  # expected value is calculated by onnx-runtime
    output = run_node(node_def, [data, roi, scales, sizes])
    np.testing.assert_allclose(output['Y'], expected, rtol=1e-6, atol=1e-6)

    # resize_linear_half_pixel_scales
    node_def = helper.make_node("Resize",
                                inputs=['X', 'roi', 'scales'],
                                outputs=['Y'],
                                mode='linear')
    scales = np.array([1, 1, 0.8, 0.8], dtype=np.float32)
    expected = np.array(
        [[[[2.375, 3.625, 4.875, 6.125, 7.375, 8.625, 9.875, 11.125],
           [14.875, 16.125, 17.375, 18.625, 19.875, 21.125, 22.375, 23.625],
           [27.375, 28.625, 29.875, 31.125, 32.375, 33.625, 34.875, 36.125],
           [39.875, 41.125, 42.375, 43.625, 44.875, 46.125, 47.375, 48.625],
           [52.375, 53.625, 54.875, 56.125, 57.375, 58.625, 59.875, 61.125],
           [64.875, 66.125, 67.375, 68.625, 69.875, 71.125, 72.375, 73.625],
           [77.375, 78.625, 79.875, 81.125, 82.375, 83.625, 84.875, 86.125],
           [89.875, 91.125, 92.375, 93.625, 94.875, 96.125, 97.375, 98.625]]]],
        dtype=np.float32)  # expected value is calculated by onnx-runtime
    output = run_node(node_def, [data, roi, scales])
    np.testing.assert_almost_equal(output["Y"], expected)

    # resize_linear_half_pixel_sizes
    node_def = helper.make_node("Resize",
                                inputs=['X', 'roi', 'scales', 'sizes'],
                                outputs=['Y'],
                                mode='linear')
    scales = np.array([], dtype=np.float32)
    sizes = np.array([1, 1, 7, 7], dtype=np.int64)
    expected = np.array([[[[
        3.357143, 4.785714, 6.214286, 7.642857, 9.071428, 10.5, 11.928572
    ], [
        17.642857, 19.071428, 20.5, 21.92857, 23.357141, 24.785713, 26.214285
    ], [
        31.928572, 33.357143, 34.785713, 36.214287, 37.642857, 39.071426, 40.5
    ], [
        46.214287, 47.642857, 49.071426, 50.5, 51.928574, 53.357143, 54.785713
    ], [60.5, 61.928577, 63.357147, 64.78572, 66.21429, 67.64286, 69.07143
       ], [
           74.78572, 76.21429, 77.64285, 79.07143, 80.50001, 81.92857, 83.35715
       ], [89.07143, 90.5, 91.928566, 93.35715, 94.78571, 96.21428, 97.64285]]]
                        ],
                        dtype=np.float32
                       )  # expected value is calculated by onnx-runtime
    output = run_node(node_def, [data, roi, scales, sizes])
    np.testing.assert_allclose(output['Y'], expected, rtol=1e-6, atol=1e-6)

    # resize_cubic_align_corners_scales
    node_def = helper.make_node("Resize",
                                inputs=['X', 'roi', 'scales'],
                                outputs=['Y'],
                                coordinate_transformation_mode="align_corners",
                                mode='cubic',
                                cubic_coeff_a=-0.5,
                                exclude_outside=1)
    scales = np.array([1, 1, 0.8, 0.8], dtype=np.float32)
    expected = np.array(
        [[[[
            1., 2.285714, 3.5714293, 4.857143, 6.142857, 7.4285717, 8.714287,
            10.
        ],
           [
               13.857139, 15.142854, 16.42857, 17.714281, 18.999994, 20.28571,
               21.571426, 22.857138
           ],
           [
               26.71429, 28.000004, 29.285723, 30.571436, 31.85715, 33.142864,
               34.428577, 35.71429
           ],
           [
               39.57143, 40.857143, 42.142864, 43.428574, 44.714287, 46.000004,
               47.28572, 48.57143
           ],
           [
               52.42857, 53.714287, 55., 56.285717, 57.571423, 58.857143,
               60.14286, 61.42857
           ],
           [
               65.28571, 66.57144, 67.85715, 69.14285, 70.428566, 71.71429,
               73.00001, 74.28571
           ],
           [
               78.14287, 79.42857, 80.7143, 82.00001, 83.28573, 84.571434,
               85.857155, 87.14287
           ],
           [
               91., 92.28572, 93.57144, 94.85715, 96.14285, 97.42858, 98.714294,
               100.
           ]]]],
        dtype=np.float32)  # expected value is calculated by onnx-runtime
    output = run_node(node_def, [data, roi, scales])
    np.testing.assert_allclose(output['Y'], expected, rtol=1e-1, atol=1e-6)

    # resize_cubic_align_corners_sizes
    node_def = helper.make_node("Resize",
                                inputs=['X', 'roi', 'scales', 'sizes'],
                                outputs=['Y'],
                                coordinate_transformation_mode="align_corners",
                                mode='cubic',
                                cubic_coeff_a=-0.5,
                                exclude_outside=1)
    scales = np.array([], dtype=np.float32)
    sizes = np.array([1, 1, 7, 7], dtype=np.int64)
    expected = np.array(
        [[[[1., 2.5, 4., 5.5, 7., 8.5, 10.],
           [16., 17.5, 19., 20.5, 22., 23.5, 25.],
           [31., 32.5, 34., 35.5, 37., 38.5, 40.],
           [46., 47.5, 49., 50.5, 52., 53.5, 55.],
           [61., 62.5, 64., 65.5, 67., 68.5, 70.],
           [76., 77.5, 79., 80.5, 82., 83.5, 85.],
           [91., 92.5, 94., 95.5, 97., 98.5, 100.]]]],
        dtype=np.float32)  # expected value is calculated by onnx-runtime
    output = run_node(node_def, [data, roi, scales, sizes])
    np.testing.assert_almost_equal(output["Y"], expected)

    # resize_cubic_asymmetric_scales
    node_def = helper.make_node("Resize",
                                inputs=['X', 'roi', 'scales'],
                                outputs=['Y'],
                                coordinate_transformation_mode="asymmetric",
                                mode='cubic',
                                cubic_coeff_a=-0.5,
                                exclude_outside=1)
    scales = np.array([1, 1, 0.8, 0.8], dtype=np.float32)
    expected = np.array(
        [[[[1., 2.25, 3.5, 4.75, 6., 7.25, 8.5, 9.832117],
           [13.5, 14.75, 16., 17.25, 18.5, 19.75, 21., 22.332117],
           [26., 27.25, 28.5, 29.75, 31., 32.25, 33.5, 34.83212],
           [38.5, 39.75, 41., 42.25, 43.5, 44.75, 46., 47.332115],
           [51., 52.25, 53.5, 54.75, 56., 57.25, 58.5, 59.83212],
           [63.5, 64.75, 66., 67.25, 68.5, 69.75, 71., 72.332115],
           [76., 77.25, 78.5, 79.75, 81., 82.25, 83.5, 84.832115],
           [
               89.32117, 90.57117, 91.82117, 93.07117, 94.32117, 95.57117,
               96.82117, 98.15329
           ]]]],
        dtype=np.float32)  # expected value is calculated by onnx-runtime
    output = run_node(node_def, [data, roi, scales])
    np.testing.assert_allclose(output['Y'], expected, rtol=1e-1, atol=1e-6)

    # resize_cubic_asymmetric_sizes
    node_def = helper.make_node("Resize",
                                inputs=['X', 'roi', 'scales', 'sizes'],
                                outputs=['Y'],
                                coordinate_transformation_mode="asymmetric",
                                mode='cubic',
                                cubic_coeff_a=-0.5,
                                exclude_outside=1)
    scales = np.array([], dtype=np.float32)
    sizes = np.array([1, 1, 7, 7], dtype=np.int64)
    expected = np.array(
        [[[[1., 2.4285715, 3.8571432, 5.285714, 6.7142863, 8.142857, 9.66485],
           [
               15.285712, 16.714287, 18.142855, 19.571426, 21., 22.428568,
               23.950563
           ],
           [
               29.57143, 31.000004, 32.428574, 33.857147, 35.285713, 36.714283,
               38.236282
           ],
           [
               43.857143, 45.285717, 46.714287, 48.142868, 49.571434, 50.999992,
               52.52199
           ],
           [
               58.142864, 59.57144, 61.000004, 62.428585, 63.857155, 65.28572,
               66.80771
           ],
           [
               72.428566, 73.85715, 75.28572, 76.714294, 78.14285, 79.57143,
               81.09343
           ],
           [
               87.6485, 89.07708, 90.505646, 91.93422, 93.36279, 94.79135,
               96.313354
           ]]]],
        dtype=np.float32)  # expected value is calculated by onnx-runtime
    output = run_node(node_def, [data, roi, scales, sizes])
    np.testing.assert_allclose(output['Y'], expected, rtol=1e-1, atol=1e-6)

    # resize_cubic_half_pixel_scales
    node_def = helper.make_node("Resize",
                                inputs=['X', 'roi', 'scales'],
                                outputs=['Y'],
                                mode='cubic',
                                cubic_coeff_a=-0.5,
                                exclude_outside=1)
    scales = np.array([1, 1, 0.8, 0.8], dtype=np.float32)
    expected = np.array([[
        [[
            1.8098788, 3.1112535, 4.3612537, 5.6112533, 6.8612533, 8.111254,
            9.361254, 10.662629
        ],
         [14.823625, 16.125, 17.375, 18.625, 19.875, 21.125, 22.375, 23.676373],
         [27.323626, 28.625, 29.875, 31.125, 32.375, 33.625, 34.875, 36.176376],
         [39.823627, 41.125, 42.375, 43.625, 44.875, 46.125, 47.375, 48.676376],
         [52.323624, 53.625, 54.875, 56.125, 57.375, 58.625, 59.875, 61.176373],
         [64.82362, 66.125, 67.375, 68.625, 69.875, 71.125, 72.375, 73.67638],
         [77.32362, 78.625, 79.875, 81.125, 82.375, 83.625, 84.875, 86.17638],
         [
             90.33737, 91.63874, 92.88875, 94.13875, 95.38875, 96.63875,
             97.88875, 99.190125
         ]]
    ]],
                        dtype=np.float32
                       )  # expected value is calculated by onnx-runtime
    output = run_node(node_def, [data, roi, scales])
    np.testing.assert_allclose(output['Y'], expected, rtol=1e-6, atol=1e-6)

    # resize_cubic_half_pixel_sizes
    node_def = helper.make_node("Resize",
                                inputs=['X', 'roi', 'scales', 'sizes'],
                                outputs=['Y'],
                                mode='cubic',
                                cubic_coeff_a=-0.5,
                                exclude_outside=1)
    scales = np.array([], dtype=np.float32)
    sizes = np.array([1, 1, 7, 7], dtype=np.int64)
    expected = np.array([[[
        [2.52846, 4.0323663, 5.460938, 6.889509, 8.318081, 9.746653, 11.250559],
        [
            17.567522, 19.071426, 20.499996, 21.928568, 23.357141, 24.785715,
            26.28962
        ],
        [
            31.853237, 33.357143, 34.785713, 36.21429, 37.64286, 39.07143,
            40.575344
        ],
        [46.13895, 47.642857, 49.071434, 50.5, 51.928566, 53.357147, 54.861053],
        [
            60.42467, 61.92858, 63.357147, 64.78572, 66.214294, 67.64287,
            69.14677
        ],
        [
            74.710396, 76.214294, 77.64286, 79.07144, 80.50001, 81.92858,
            83.432495
        ],
        [89.749466, 91.253365, 92.68193, 94.1105, 95.53907, 96.96766, 98.47156]
    ]]],
                        dtype=np.float32
                       )  # expected value is calculated by onnx-runtime
    output = run_node(node_def, [data, roi, scales, sizes])
    np.testing.assert_allclose(output['Y'], expected, rtol=1e-2, atol=1e-6)

    # crop_and_resize_nearest with scales
    node_def = helper.make_node(
        "Resize",
        inputs=['X', 'roi', 'scales'],
        outputs=['Y'],
        coordinate_transformation_mode='tf_crop_and_resize',
        mode='nearest',
        nearest_mode='round_prefer_ceil',
        extrapolation_value=-20.0)
    roi = np.array([0, 0, 0.4, 0.6, 1, 1, 1.2, 1.7], dtype=np.float32)
    scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)
    expected = np.array(
        [[[[46., 48., 49., -20., -20., -20., -20., -20.],
           [56., 58., 59., -20., -20., -20., -20., -20.],
           [66., 68., 69., -20., -20., -20., -20., -20.],
           [76., 78., 79., -20., -20., -20., -20., -20.],
           [86., 88., 89., -20., -20., -20., -20., -20.],
           [96., 98., 99., -20., -20., -20., -20., -20.],
           [-20., -20., -20., -20., -20., -20., -20., -20.],
           [-20., -20., -20., -20., -20., -20., -20., -20.]]]],
        dtype=np.float32)  # expected value is calculated by onnx-runtime
    output = run_node(node_def, [data, roi, scales])
    np.testing.assert_almost_equal(output["Y"], expected)

    # crop_and_resize_nearest with sizes
    node_def = helper.make_node(
        "Resize",
        inputs=['X', 'roi', 'scales', 'sizes'],
        outputs=['Y'],
        coordinate_transformation_mode='tf_crop_and_resize',
        mode='nearest',
        nearest_mode='round_prefer_ceil',
    )
    roi = np.array([0, 0, 0.4, 0.6, 1, 1, 1.2, 1.7], dtype=np.float16)
    scales = np.array([], dtype=np.float32)
    sizes = np.array([1, 1, 7, 7], dtype=np.int64)
    expected = np.array(
        [[[[46., 48., 50., 0., 0., 0., 0.], [56., 58., 60., 0., 0., 0., 0.],
           [66., 68., 70., 0., 0., 0., 0.], [76., 78., 80., 0., 0., 0., 0.],
           [86., 88., 90., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0.]]]],
        dtype=np.float32)  # expected value is calculated by onnx-runtime
    output = run_node(node_def, [data, roi, scales, sizes])
    np.testing.assert_almost_equal(output["Y"], expected)

    # crop_and_resize_linear with scales
    node_def = helper.make_node(
        "Resize",
        inputs=['X', 'roi', 'scales'],
        outputs=['Y'],
        mode='linear',
        coordinate_transformation_mode='tf_crop_and_resize',
        extrapolation_value=20.0)
    roi = np.array([0, 0, 0.4, 0.6, 1, 1, 1.2, 1.7], dtype=np.float64)
    scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)
    expected = np.array(
        [[[[42.4, 43.814285, 45.228573, 20., 20., 20., 20., 20.],
           [52.685715, 54.100002, 55.514286, 20., 20., 20., 20., 20.],
           [62.971436, 64.38572, 65.80001, 20., 20., 20., 20., 20.],
           [73.25715, 74.67143, 76.08572, 20., 20., 20., 20., 20.],
           [83.54286, 84.957146, 86.37143, 20., 20., 20., 20., 20.],
           [93.82858, 95.24287, 96.65715, 20., 20., 20., 20., 20.],
           [20., 20., 20., 20., 20., 20., 20., 20.],
           [20., 20., 20., 20., 20., 20., 20., 20.]]]],
        dtype=np.float32)  # expected value is calculated by onnx-runtime
    # sys_config.auto_cast=False and roi_dtype=float64 should throw exception
    self.assertRaises(RuntimeError, run_node, node_def,
                      [data, roi.astype(np.float64), scales])

    # crop_and_resize_linear with sizes
    node_def = helper.make_node(
        "Resize",
        inputs=['X', 'roi', 'scales', 'sizes'],
        outputs=['Y'],
        mode='linear',
        coordinate_transformation_mode='tf_crop_and_resize',
        extrapolation_value=50.0)
    roi = np.array([0, 0, 0.4, 0.6, 1, 1, 1.2, 1.7], dtype=np.float32)
    scales = np.array([], dtype=np.float32)
    sizes = np.array([1, 1, 7, 7], dtype=np.int64)
    expected = np.array(
        [[[[42.4, 44.05, 45.7, 50., 50., 50., 50.],
           [54.4, 56.050003, 57.700005, 50., 50., 50., 50.],
           [66.40001, 68.05, 69.700005, 50., 50., 50., 50.],
           [78.40001, 80.05001, 81.700005, 50., 50., 50., 50.],
           [90.40001, 92.05, 93.70001, 50., 50., 50., 50.],
           [50., 50., 50., 50., 50., 50., 50.],
           [50., 50., 50., 50., 50., 50., 50.]]]],
        dtype=np.float32)  # expected value is calculated by onnx-runtime
    output = run_node(node_def, [data, roi, scales, sizes])
    np.testing.assert_allclose(output["Y"], expected, rtol=1e-6, atol=1e-6)

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
      node_def = helper.make_node("Pad", ["X"], ["Y"],
                                  mode="constant",
                                  pads=[1, 1, 1, 1],
                                  value=2.0)
      output = run_node(node_def, [x])
      y = np.pad(x, ((1, 1), (1, 1)), 'constant', constant_values=(2, 2))
      np.testing.assert_almost_equal(output["Y"], y)
      # mode = reflect and edge
      for mode in ['edge', 'reflect']:
        node_def = helper.make_node("Pad", ["X"], ["Y"],
                                    mode=mode,
                                    pads=[1, 1, 1, 1])
        output = run_node(node_def, [x])
        y = np.pad(x, ((1, 1), (1, 1)), mode)
        np.testing.assert_almost_equal(output["Y"], y)
    else:  # for opset >= 11
      # mode = constant
      node_def = helper.make_node("Pad", ["X", "pads", "constant_values"],
                                  ["Y"],
                                  mode="constant")
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
      # negative pads
      node_def = helper.make_node("Pad", ["X", "pads"], ["Y"], mode="constant")
      pads = np.array([-2, -2, -2, -2], dtype=np.int64)
      x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape((3, 4))
      y = x
      x = np.pad(x, ((2, 2), (2, 2)), 'constant')
      output = run_node(node_def, [x, pads])
      np.testing.assert_almost_equal(output["Y"], y)

      # negative pads with 3 dimensions
      node_def = helper.make_node("Pad", ["X", "pads"], ["Y"], mode="constant")
      pads = np.array([-1, 0, 0, 0, -1, 0], dtype=np.int64)
      x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape((2, 3, 2))
      y = np.array([7, 8, 9, 10]).reshape((1, 2, 2))
      output = run_node(node_def, [x, pads])
      np.testing.assert_almost_equal(output["Y"], y)

      if legacy_opset_pre_ver(13) is False:  # for opset = 13
        # data type = string, with specific constant_value
        node_def = helper.make_node("Pad", ["X", "pads", "constant_values"],
                                    ["Y"],
                                    mode="constant")
        x = np.chararray((10, 10), itemsize=2)
        a1 = 'a1'
        b1 = 'b1'
        blank = ''
        x[:] = a1.encode('UTF-8')
        pads = np.array([1, 1, 1, 1], dtype=np.int64)
        constant_values = b1.encode('UTF-8')
        y = np.pad(x, ((1, 1), (1, 1)),
                   'constant',
                   constant_values=(b1.encode('UTF-8'), b1.encode('UTF-8')))
        output = run_node(node_def, [x, pads, constant_values])
        np.testing.assert_array_equal(output["Y"], y)

        # data type = string, with default constant_value
        node_def = helper.make_node("Pad", ["X", "pads"], ["Y"],
                                    mode="constant")
        y = np.pad(x, ((1, 1), (1, 1)),
                   'constant',
                   constant_values=(blank.encode('UTF-8'),
                                    blank.encode('UTF-8')))
        output = run_node(node_def, [x, pads])
        np.testing.assert_array_equal(output["Y"], y)

  def test_qlinearconv(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support QLinearConv.".format(
              defs.onnx_opset_version()))
    for device in self._get_device_list():
      # Test w_scale and w_zero_point as scalar
      node_def = helper.make_node("QLinearConv",
                                  inputs=[
                                      "x", "x_scale", "x_zero_point", "w",
                                      "w_scale", "w_zero_point", "y_scale",
                                      "y_zero_point"
                                  ],
                                  outputs=["Y"])
      x = np.array([
          [255, 174, 162, 25, 203, 168, 58],
          [15, 59, 237, 95, 129, 0, 64],
          [56, 242, 153, 221, 168, 12, 166],
          [232, 178, 186, 195, 237, 162, 237],
          [188, 39, 124, 77, 80, 102, 43],
          [127, 230, 21, 83, 41, 40, 134],
          [255, 154, 92, 141, 42, 148, 247],
      ],
                   dtype=np.uint8).reshape((1, 1, 7, 7))
      x_scale = np.float32(0.00369204697)
      x_zero_point = np.uint8(132)

      w = np.array([0], dtype=np.uint8).reshape((1, 1, 1, 1))
      w_scale = np.float32(0.00172794575)
      w_zero_point = np.uint8(255)

      y = np.array([
          [0, 81, 93, 230, 52, 87, 197],
          [240, 196, 18, 160, 126, 255, 191],
          [199, 13, 102, 34, 87, 243, 89],
          [23, 77, 69, 60, 18, 93, 18],
          [67, 216, 131, 178, 175, 153, 212],
          [128, 25, 234, 172, 214, 215, 121],
          [0, 101, 163, 114, 213, 107, 8],
      ],
                   dtype=np.uint8).reshape((1, 1, 7, 7))
      y_scale = np.float32(0.00162681262)
      y_zero_point = np.uint8(123)

      output = run_node(node_def, [
          x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale,
          y_zero_point
      ],
                        device=device)
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

    # test tensor(uint8), tensor(int8)
    node_def = helper.make_node("ReduceMax", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd_int(0, 100, [5, 10, 10, 3], np.uint8)
    output = run_node(node_def, [x])
    np.testing.assert_allclose(output["Y"],
                               np.max(x, (1, 2), keepdims=True),
                               rtol=1e-3)

    node_def = helper.make_node("ReduceMax", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd_int(-100, 100, [5, 10, 10, 3], np.int8)
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

    # test tensor(uint8), tensor(int8)
    node_def = helper.make_node("ReduceMin", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd_int(0, 100, [5, 10, 10, 3], np.uint8)
    output = run_node(node_def, [x])
    np.testing.assert_allclose(output["Y"],
                               np.min(x, (1, 2), keepdims=True),
                               rtol=1e-3)

    node_def = helper.make_node("ReduceMin", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd_int(-100, 100, [5, 10, 10, 3], np.int8)
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
    x = self._get_rnd_float32(shape=[5, 10, 10, 3])
    axes = np.array([1, 2], dtype=np.int64)
    if legacy_opset_pre_ver(13):  # for opset 1 to 12
      node_def = helper.make_node("ReduceSum", ["X"], ["Y"], axes=axes)
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

  def _run_scan_node(self,
                     initial,
                     x1,
                     x2,
                     input_shape,
                     output_shape,
                     scan_input_axes=None,
                     scan_input_directions=None,
                     scan_output_axes=None,
                     scan_output_directions=None,
                     sequence_lens=None,
                     directions=None):
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
                                   ["concat_out"],
                                   axis=0)
    add_node = helper.make_node("Add", ["concat_out", "const_1"], ["add_out"])
    split_node = helper.make_node(
        "Split", ["add_out"],
        ["split1_out", "split2_out", "split3_out", "split4_out"])

    state_in = helper.make_tensor_value_info('state_in', TensorProto.FLOAT, [1])
    concat1_in = helper.make_tensor_value_info('concat1_in', TensorProto.FLOAT,
                                               input_shape)
    concat2_in = helper.make_tensor_value_info('concat2_in', TensorProto.FLOAT,
                                               input_shape)
    state_out = helper.make_tensor_value_info('state_out', TensorProto.FLOAT,
                                              [1])
    split1_out = helper.make_tensor_value_info('split1_out', TensorProto.FLOAT,
                                               output_shape)
    split2_out = helper.make_tensor_value_info('split2_out', TensorProto.FLOAT,
                                               output_shape)
    split3_out = helper.make_tensor_value_info('split3_out', TensorProto.FLOAT,
                                               output_shape)
    split4_out = helper.make_tensor_value_info('split4_out', TensorProto.FLOAT,
                                               output_shape)

    scan_body = helper.make_graph(
        [constant_node, state_add_node, concat_node, add_node, split_node],
        "scan_body",
        [state_in, concat1_in, concat2_in],
        [state_out, split1_out, split2_out, split3_out, split4_out],
    )

    node_kwargs = {
        "op_type": "Scan",
        "inputs": ["initial", "x1", "x2"],
        "outputs": ["y", "z1", "z2", "z3", "z4"],
        "num_scan_inputs": 2,
        "body": scan_body
    }
    if sequence_lens is not None:
      node_kwargs["inputs"] = ["" if sequence_lens is str else "seq_lens"
                              ] + node_kwargs["inputs"]

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
      raise unittest.SkipTest("ONNX version {} not supported.".format(
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
    x2_out = x2[:, ::-1] + 1

    Z = np.concatenate([x1_out, x2_out], 2)
    if sequence_lens is not str:
      for batch in range(len(sequence_lens)):
        # zero pad from the sequence_lens
        shape = list(np.shape(Z[batch]))
        seq_len = sequence_lens[batch]

        zero_pad = np.zeros([shape[0] - seq_len] + shape[1:])
        Z[batch] = np.concatenate([Z[batch][:seq_len], zero_pad])

    output = self._run_scan_node(initial,
                                 x1,
                                 x2, [6, 4], [3, 2],
                                 sequence_lens=sequence_lens,
                                 directions=directions)
    output_z = np.concatenate(
        [output["z1"], output["z2"], output["z3"], output["z4"]], 2)

    np.testing.assert_almost_equal(output["y"], Y)
    np.testing.assert_almost_equal(output_z, Z)

  def test_scan(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} not supported.".format(
          defs.onnx_opset_version()))

    initial = self._get_rnd_int(0, 100, shape=[2]).astype(np.float32)
    x1 = self._get_rnd_float32(0, 1000, shape=[20, 6, 2])
    x2 = self._get_rnd_float32(0, 1000, shape=[20, 6, 2])

    Y = initial + np.shape(x1)[0]
    Z = np.concatenate([x1, x2], 1) + 1

    output = self._run_scan_node(initial, x1, x2, [6, 2], [3, 2])
    output_z = np.concatenate(
        [output["z1"], output["z2"], output["z3"], output["z4"]], 1)

    np.testing.assert_almost_equal(output["y"], Y)
    np.testing.assert_almost_equal(output_z, Z)

  def test_scan_input_directions(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} not supported.".format(
          defs.onnx_opset_version()))

    initial = self._get_rnd_int(0, 100, shape=[1]).astype(np.float32)
    x1 = self._get_rnd_float32(0, 1000, shape=[20, 6, 2])
    x2 = self._get_rnd_float32(0, 1000, shape=[20, 6, 2])

    Y = initial + np.shape(x1)[0]
    Z = np.concatenate([x1[::-1], x2], 1) + 1

    output = self._run_scan_node(initial,
                                 x1,
                                 x2, [6, 2], [3, 2],
                                 scan_input_directions=[1, 0])
    output_z = np.concatenate(
        [output["z1"], output["z2"], output["z3"], output["z4"]], 1)

    np.testing.assert_almost_equal(output["y"], Y)
    np.testing.assert_almost_equal(output_z, Z)

  def test_scan_input_axes(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} not supported.".format(
          defs.onnx_opset_version()))

    initial = self._get_rnd_int(0, 100, shape=[1]).astype(np.float32)
    x1 = self._get_rnd_float32(0, 1000, shape=[20, 6, 2])
    x2 = self._get_rnd_float32(0, 1000, shape=[20, 6, 2])

    Y = initial + np.shape(x1)[1]
    x1_transpose = np.transpose(x1, (1, 0, 2))
    x2_transpose = np.transpose(x2, (1, 0, 2))
    Z = np.concatenate([x1_transpose, x2_transpose], 1) + 1

    output = self._run_scan_node(initial,
                                 x1,
                                 x2, [3, 2], [10, 2],
                                 scan_input_axes=[1, 1])
    output_z = np.concatenate(
        [output["z1"], output["z2"], output["z3"], output["z4"]], 1)

    np.testing.assert_almost_equal(output["y"], Y)
    np.testing.assert_almost_equal(output_z, Z)

  def test_scan_output_directions(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} not supported.".format(
          defs.onnx_opset_version()))

    initial = self._get_rnd_int(0, 100, shape=[1]).astype(np.float32)
    x1 = self._get_rnd_float32(0, 1000, shape=[20, 6, 2])
    x2 = self._get_rnd_float32(0, 1000, shape=[20, 6, 2])

    Y = initial + np.shape(x1)[0]
    Z = np.concatenate([x1, x2], 1) + 1

    output = self._run_scan_node(initial,
                                 x1,
                                 x2, [6, 2], [3, 2],
                                 scan_output_directions=[1, 0, 0, 1])
    output_z = np.concatenate(
        [output["z1"][::-1], output["z2"], output["z3"], output["z4"][::-1]], 1)

    np.testing.assert_almost_equal(output["y"], Y)
    np.testing.assert_almost_equal(output_z, Z)

  def test_scan_output_axes(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} not supported.".format(
          defs.onnx_opset_version()))

    initial = self._get_rnd_int(0, 100, shape=[1]).astype(np.float32)
    x1 = self._get_rnd_float32(0, 1000, shape=[20, 6, 2])
    x2 = self._get_rnd_float32(0, 1000, shape=[20, 6, 2])

    Y = initial + np.shape(x1)[0]
    Z = np.concatenate([x1, x2], 1) + 1
    Z = np.transpose(Z, (1, 0, 2))

    output = self._run_scan_node(initial,
                                 x1,
                                 x2, [10, 2], [3, 2],
                                 scan_output_axes=[1, 1, 1, 1])
    output_z = np.concatenate(
        [output["z1"], output["z2"], output["z3"], output["z4"]], 0)

    np.testing.assert_almost_equal(output["y"], Y)
    np.testing.assert_almost_equal(output_z, Z)

  def test_scatter_elements1(self):
    data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    indices = np.array([[1, 3]], dtype=np.int64)
    updates = np.array([[1.1, 2.1]], dtype=np.float32)
    axis = 1
    ref_output = np.array([[1.0, 1.1, 3.0, 2.1, 5.0]], dtype=np.float32)

    if legacy_opset_pre_ver(11):
      node_def = helper.make_node("Scatter", ["data", "indices", "updates"],
                                  ["outputs"],
                                  axis=axis)
      output = run_node(node_def, [data, indices, updates])
      np.testing.assert_almost_equal(output["outputs"], ref_output)
    else:
      node_def = helper.make_node("ScatterElements",
                                  ["data", "indices", "updates"], ["outputs"],
                                  axis=axis)
      output = run_node(node_def, [data, indices, updates])
      np.testing.assert_almost_equal(output["outputs"], ref_output)

      # test data types that are not natively supported by Tensorflow
      self.assertRaises(RuntimeError, run_node, node_def,
                        [np.complex64(data), indices,
                         np.complex64(updates)])

  def test_scatter_elements2(self):
    data = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ],
                    dtype=np.float32)
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
    ],
                          dtype=np.float32)

    if legacy_opset_pre_ver(11):
      node_def = helper.make_node("Scatter", ["data", "indices", "updates"],
                                  ["outputs"])
      output = run_node(node_def, [data, indices, updates])
      np.testing.assert_almost_equal(output["outputs"], ref_output)
    else:
      node_def = helper.make_node("ScatterElements",
                                  ["data", "indices", "updates"], ["outputs"])
      output = run_node(node_def, [data, indices, updates])
      np.testing.assert_almost_equal(output["outputs"], ref_output)

  def test_scatter_elements3(self):
    # indices out of bounds
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.int64)
    updates = np.array([[1.1, 2.1, 3.1]], dtype=np.float32)

    if legacy_opset_pre_ver(11):
      node_def = helper.make_node("Scatter", ["data", "indices", "updates"],
                                  ["outputs"])
    else:
      node_def = helper.make_node("ScatterElements",
                                  ["data", "indices", "updates"], ["outputs"])
    with np.testing.assert_raises(tf.errors.InvalidArgumentError):
      run_node(node_def, [data, indices, updates])

  def test_scatter_nd(self):
    if legacy_opset_pre_ver(11):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support ScatterND.".format(
              defs.onnx_opset_version()))

    # valid positve and negative indices for elements
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    indices = np.array([[4], [3], [1], [7]], dtype=np.int64)
    updates = np.array([9, 10, 11, 12], dtype=np.float32)
    ref_output = np.array([1, 11, 3, 10, 9, 6, 7, 12], dtype=np.float32)
    node_def = helper.make_node("ScatterND", ["data", "indices", "updates"],
                                ["outputs"])
    output = run_node(node_def, [data, indices, updates])
    np.testing.assert_almost_equal(output["outputs"], ref_output)

    # valid positive and negative indices for slices
    data = np.reshape(np.arange(1, 25, dtype=np.float32), [2, 3, 4])
    indices = np.array([[-2, -1], [1, 0]], dtype=np.int64)
    updates = np.array([[39, 40, 41, 42], [43, 44, 45, 46]], dtype=np.float32)
    ref_output = np.array(
        [[[1, 2, 3, 4], [5, 6, 7, 8], [39, 40, 41, 42]],
         [[43, 44, 45, 46], [17, 18, 19, 20], [21, 22, 23, 24]]],
        dtype=np.float32)
    output = run_node(node_def, [data, indices, updates])
    np.testing.assert_almost_equal(output["outputs"], ref_output)
    indices = np.array([[-1]], dtype=np.int64)
    updates = np.array([[[43, 44, 45, 46], [47, 48, 49, 50], [51, 52, 53, 54]]],
                       dtype=np.float32)
    ref_output = np.array(
        [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
         [[43, 44, 45, 46], [47, 48, 49, 50], [51, 52, 53, 54]]],
        dtype=np.float32)
    output = run_node(node_def, [data, indices, updates])
    np.testing.assert_almost_equal(output["outputs"], ref_output)

    # indices out of bounds
    indices = np.array([[0, 1, 2], [-1, -1, -3], [-2, -3, -4], [0, 2, -5]],
                       dtype=np.int64)
    updates = np.array([37, 52, 30, 39], dtype=np.float32)
    with np.testing.assert_raises(tf.errors.InvalidArgumentError):
      run_node(node_def, [data, indices, updates])
    indices = np.array([[0, 1], [-1, -1], [-2, -4]], dtype=np.int64)
    updates = np.array([[35, 36, 37, 38], [51, 52, 53, 54], [31, 32, 33, 34]],
                       dtype=np.float32)
    with np.testing.assert_raises(tf.errors.InvalidArgumentError):
      run_node(node_def, [data, indices, updates])

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

    # test to drive data cast code path
    node_def = helper.make_node("Sign", ["X"], ["Y"])
    x = np.array([[-10, 5, 3], [8, -6, 7], [8, 6, -7]]).astype(np.int16)
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
    axes = np.array([0, 1, 2])
    starts = np.array([0, 0, 0])
    ends = np.array([2, 2, 2])
    steps = np.array([1, 1, 1])

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
    axes = np.array([0, 2])
    starts = np.array([0, -7])
    ends = np.array([-8, 20])

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
    axes = np.array([0, 1, 2])
    starts = np.array([0, 0, 0])
    ends = np.array([2, 2, 2])
    steps = np.array([2, -2, -1])

    if legacy_opset_pre_ver(10) == False:
      node_def = helper.make_node("Slice",
                                  ["X", "starts", "ends", "axes", "steps"],
                                  ["S"])
      x = self._get_rnd_float32(shape=[1000]).reshape([10, 10, 10])
      output = run_node(node_def, [x, starts, ends, axes, steps])
      np.testing.assert_almost_equal(output["S"], x[0:2:2, 0:2:-2, 0:2:-1])

  def test_softmax(self):
    node_def = helper.make_node("Softmax", ["X"], ["Y"], axis=0)
    x = self._get_rnd_float32(shape=[3, 4, 5])
    output = run_node(node_def, [x])
    if legacy_opset_pre_ver(13):  # opset 1 & 11
      x = x.reshape(1, 60)
      max_x = np.max(x, axis=1).reshape((-1, 1))
      exp_x = np.exp(x - max_x)
      y = exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))
      y = y.reshape(3, 4, 5)
    else:  # opset 13
      x_max = np.max(x, axis=0, keepdims=True)
      tmp = np.exp(x - x_max)
      s = np.sum(tmp, axis=0, keepdims=True)
      y = tmp / s
    np.testing.assert_almost_equal(output["Y"], y)

  def test_softplus(self):
    node_def = helper.make_node("Softplus", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[3, 4, 5])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"],
                                   np.log(np.exp(x) + 1),
                                   decimal=5)

  def test_softsign(self):
    node_def = helper.make_node("Softsign", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[3, 4, 5])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], x / (1 + np.abs(x)))

  def test_space_to_depth(self):
    for device in self._get_device_list():
      node_def = helper.make_node("SpaceToDepth", ["X"], ["Y"], blocksize=2)
      x_shape = [1, 3, 2, 2]
      x = self._get_rnd_float32(shape=x_shape)
      output = run_node(node_def, [x], device=device)
      x = np.transpose(x, (0, 2, 3, 1))
      y = np.reshape(np.swapaxes(x.reshape(1, 1, 1, 1, 1, 12), 2, 3),
                     (1, 1, 1, 12))
      y = np.transpose(y, (0, 3, 1, 2))
      np.testing.assert_allclose(output["Y"], y, rtol=1e-3)
      # test data type that is not natively supported by Tensorflow GPU
      x = self._get_rnd_float32(shape=x_shape).astype(np.float64)
      output = run_node(node_def, [x], device=device)
      x = np.transpose(x, (0, 2, 3, 1))
      y = np.reshape(np.swapaxes(x.reshape(1, 1, 1, 1, 1, 12), 2, 3),
                     (1, 1, 1, 12))
      y = np.transpose(y, (0, 3, 1, 2))
      np.testing.assert_allclose(output["Y"], y, rtol=1e-3)

  def test_split(self):
    split = np.array([3, 3, 4]).astype(np.int64)
    x = self._get_rnd_float32(shape=[100]).reshape([10, 10])
    if legacy_opset_pre_ver(13):  # for opset 1 to 12
      node_def = helper.make_node("Split", ["X"],
                                  ["Z%i" % i for i in range(len(split))],
                                  axis=0,
                                  split=split)
      output = run_node(node_def, [x])
    else:  # for opset 13 or above
      node_def = helper.make_node("Split", ["X", "split"],
                                  ["Z%i" % i for i in range(len(split))],
                                  axis=0)
      output = run_node(node_def, [x, split])
    for a, b in zip(list(output), np.split(x, np.cumsum(split))[:-1]):
      np.testing.assert_almost_equal(a, b)

    # test axis out of bound
    if legacy_opset_pre_ver(13):  # for opset 1 to 12
      node_def = helper.make_node("Split", ["X"],
                                  ["Z%i" % i for i in range(len(split))],
                                  axis=3,
                                  split=split)
      self.assertRaises(ValueError, run_node, node_def, [x])
    else:  # for opset 13 or above
      node_def = helper.make_node("Split", ["X", "split"],
                                  ["Z%i" % i for i in range(len(split))],
                                  axis=3)
      self.assertRaises(ValueError, run_node, node_def, [x, split])

  def test_sqrt(self):
    node_def = helper.make_node("Sqrt", ["X"], ["Y"])
    x = self._get_rnd_float32(shape=[1000]) + 1.0
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.sqrt(x), decimal=5)

  def test_squeeze(self):
    x = np.array([[[0], [1], [2]]])
    axes = np.array([2], dtype=np.int64)
    if legacy_opset_pre_ver(13):  # for opset 1 to 12
      node_def = helper.make_node("Squeeze", ["X"], ["Y"], axes=axes)
      output = run_node(node_def, [x])
      np.testing.assert_almost_equal(output["Y"], np.squeeze(x, axis=2))

  def test_sub(self):
    node_def = helper.make_node("Sub", ["X", "Y"], ["Z"])
    x = self._get_rnd_float32(shape=[10, 10])
    y = self._get_rnd_float32(shape=[10, 10])
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.subtract(x, y))
    # sys_config.auto_cast=False and x or y dtype=uint64 should throw exception
    x = self._get_rnd_int(0, 3000, [10, 10], np.uint64)
    y = self._get_rnd_int(0, 1000, [10, 10], np.uint64)
    self.assertRaises(RuntimeError, run_node, node_def, [x, y])

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
      raise unittest.SkipTest(
          "ONNX version {} doesn't support TfIdfVectorizer.".format(
              defs.onnx_opset_version()))

    def run_test_ints():
      node_def = helper.make_node("TfIdfVectorizer", ["X"], ["Y"],
                                  mode=mode,
                                  min_gram_length=min_gram_len,
                                  max_gram_length=max_gram_len,
                                  max_skip_count=max_skip,
                                  ngram_counts=ngram_counts,
                                  ngram_indexes=ngram_indexes,
                                  weights=weights,
                                  pool_int64s=pool_int64s)
      output = run_node(node_def, [x])
      np.testing.assert_almost_equal(output["Y"], y)

    def run_test_strings():
      node_def = helper.make_node("TfIdfVectorizer", ["X"], ["Y"],
                                  mode=mode,
                                  min_gram_length=min_gram_len,
                                  max_gram_length=max_gram_len,
                                  max_skip_count=max_skip,
                                  ngram_counts=ngram_counts,
                                  ngram_indexes=ngram_indexes,
                                  weights=weights,
                                  pool_strings=pool_strings)
      output = run_node(node_def, [x])
      np.testing.assert_almost_equal(output["Y"], y)

    # test 2d inputs with 3 elements, output contains 1-grams and 2-grams
    x = np.array([[1, 1, 3, 3, 3, 7], [8, 6, 7, 5, 6, 8], [8, 6, 7, 5, 6,
                                                           8]]).astype(np.int32)
    y = np.array([[0., 3., 0., 0., 0., 0., 0.], [0., 0., 1., 0., 1., 0., 1.],
                  [0., 0., 1., 0., 1., 0., 1.]]).astype(np.float32)
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
    y = np.array([[0., 0.1, 0., 0., 0., 0., 0.],
                  [0., 0., 0.1, 0., 0.5, 0.5, 0.5]]).astype(np.float32)
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
    # test data types that are not natively supported by Tensorflow
    x = self._get_rnd_int(0, 100, shape=[3, 5, 5, 3], dtype=np.uint16)
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
      values = np.array([[2, 3], [7, 10], [3, 5]], dtype=np.float32)
      indices = np.array([[1, 0], [3, 2], [3, 2]], dtype=np.int64)
      k = np.array([2], dtype=np.int64)
      node_def = helper.make_node("TopK", ["x", "k"], ["values", "indices"],
                                  largest=0,
                                  sorted=1)
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


if __name__ == '__main__':
  unittest.main()

