from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import unittest

from onnx_tf.backend import onnx_graph_to_tensorflow_rep
from onnx_tf.common.legacy import legacy_opset_pre_ver
from onnx_tf.common.pooling_helper import py_pool
from onnx import defs
from onnx import helper
from onnx import TensorProto

# Run the following test in graph mode
tf.compat.v1.disable_eager_execution()


class TestDynamicShape(unittest.TestCase):
  """ Tests for dynamic shape support
  """

  def _get_rnd_float32(self, low=-1.0, high=1.0, shape=None):
    output = np.random.uniform(low, high, shape)
    if shape is None:
      return np.float32(output)
    else:
      return output.astype(np.float32)

  def _get_rnd_int(self, low, high=None, shape=None, dtype=np.int32):
    return np.random.randint(low, high, size=shape, dtype=dtype)

  def test_arg_max(self):
    if legacy_opset_pre_ver(12):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support select_last_index attribute for ArgMax that depends on shape."
          .format(defs.onnx_opset_version()))
    axis = 1
    node_def = helper.make_node("ArgMax",
                                inputs=['X'],
                                outputs=['Y'],
                                axis=axis,
                                keepdims=0,
                                select_last_index=1)
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        ])
    x = np.array([[1, 2, 3, 5, 3, 4, 5, 1], [2, 9, 3, 5, 9, 4, 5, 1]])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    output = tf_rep.run({"X": x})
    expected_output = np.argmax(np.flip(x, axis), axis=axis)
    expected_output = x.shape[axis] - expected_output - 1
    np.testing.assert_almost_equal(output['Y'], expected_output)

  def test_arg_min(self):
    if legacy_opset_pre_ver(12):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support select_last_index attribute for ArgMin that depends on shape."
          .format(defs.onnx_opset_version()))
    axis = 1
    node_def = helper.make_node("ArgMin",
                                inputs=['X'],
                                outputs=['Y'],
                                axis=axis,
                                keepdims=0,
                                select_last_index=1)
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        ])
    x = np.array([[1, 2, 3, 5, 3, 4, 5, 1], [2, 7, 3, 5, 2, 4, 5, 6]])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    output = tf_rep.run({"X": x})
    expected_output = np.argmin(np.flip(x, axis), axis=axis)
    expected_output = x.shape[axis] - expected_output - 1
    np.testing.assert_almost_equal(output['Y'], expected_output)

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
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                          [None, None, None, None]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [None]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [None]),
            helper.make_tensor_value_info("mean", TensorProto.FLOAT, [None]),
            helper.make_tensor_value_info("var", TensorProto.FLOAT, [None])
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT,
                                          [None, None, None, None])
        ])
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
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    output = tf_rep.run({
        "X": x,
        "scale": scale,
        "bias": bias,
        "mean": m,
        "var": v
    })
    np.testing.assert_almost_equal(output["Y"], golden, decimal=5)

  def test_conv_transpose(self):
    # test dynamic batch size on transpose of 2d convolution
    pads = [1, 1, 1, 1]
    x_shape = [1, 3, 4, 6]
    x = self._get_rnd_float32(shape=x_shape)
    weight_shape = [3, 5, 2, 2]
    weights = self._get_rnd_float32(shape=weight_shape)

    node_def = helper.make_node("ConvTranspose", ["X", "weights"], ["Y"],
                                pads=pads)
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                          [None, None, None, None]),
            helper.make_tensor_value_info("weights", TensorProto.FLOAT,
                                          [None, None, None, None])
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT,
                                          [None, None, None, None])
        ])

    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    output = tf_rep.run({"X": x, "weights": weights})

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
    x = self._get_rnd_float32(shape=input_shape)
    test_output = py_pool(x,
                          kernel_shape=kernel_shape,
                          strides=strides,
                          dilations=dilations,
                          padding=pads,
                          ceil_mode=ceil_mode,
                          pooling_type="MAX",
                          include_indices=False)
    node_def = helper.make_node(op_type="MaxPool",
                                inputs=["X"],
                                outputs=["Y"],
                                kernel_shape=kernel_shape,
                                strides=strides,
                                dilations=dilations,
                                pads=pads,
                                ceil_mode=ceil_mode)
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                          [None, None, None, None]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT,
                                          [None, None, None, None])
        ])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    output = tf_rep.run({"X": x})

    np.testing.assert_almost_equal(output["Y"], test_output)

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

    input_shape = [10, 3, 23, 23]
    x = self._get_rnd_float32(shape=input_shape) - 2

    node_def = helper.make_node("MaxPool", ["X"], ["Y", "Ind"],
                                kernel_shape=kernel_shape,
                                strides=strides,
                                dilations=dilations,
                                pads=pads,
                                ceil_mode=ceil_mode)

    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                          [None, None, None, None]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT,
                                          [None, None, None, None]),
            helper.make_tensor_value_info("Ind", TensorProto.INT64,
                                          [None, None, None, None])
        ])

    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    output = tf_rep.run({"X": x})

    test_output, test_ind = py_pool(x,
                                    kernel_shape=kernel_shape,
                                    strides=strides,
                                    dilations=dilations,
                                    padding=pads,
                                    ceil_mode=ceil_mode,
                                    pooling_type="MAX")

    np.testing.assert_almost_equal(output["Y"], test_output)
    np.testing.assert_almost_equal(output["Ind"], test_ind)

  def test_slice(self):
    # test case 1 with normal inputs
    axes = [0, 1, 2]
    starts = [0, 0, 0]
    ends = [2, 2, 2]

    if legacy_opset_pre_ver(10):
      node_def = helper.make_node("Slice", ["X"], ["S"],
                                  axes=axes,
                                  starts=starts,
                                  ends=ends)
      graph_def = helper.make_graph(
          [node_def],
          name="test_unknown_shape",
          inputs=[
              helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                            [None, None, None])
          ],
          outputs=[
              helper.make_tensor_value_info("S", TensorProto.FLOAT,
                                            [None, None, None])
          ])
    else:
      node_def = helper.make_node("Slice", ["X", "starts", "ends", "axes"],
                                  ["S"])
      graph_def = helper.make_graph(
          [node_def],
          name="test_unknown_shape",
          inputs=[
              helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                            [None, None, None]),
              helper.make_tensor_value_info("starts", TensorProto.INT32,
                                            [None]),
              helper.make_tensor_value_info("ends", TensorProto.INT32, [None]),
              helper.make_tensor_value_info("axes", TensorProto.INT32, [None]),
          ],
          outputs=[
              helper.make_tensor_value_info("S", TensorProto.FLOAT,
                                            [None, None, None])
          ])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)

    if legacy_opset_pre_ver(10):
      x = self._get_rnd_float32(shape=[1000]).reshape([10, 10, 10])
      output = tf_rep.run({"X": x})
      np.testing.assert_almost_equal(output["S"], x[0:2, 0:2, 0:2])
    else:
      x = self._get_rnd_float32(shape=[1000]).reshape([10, 10, 10])
      output = tf_rep.run({
          "X": x,
          "starts": starts,
          "ends": ends,
          "axes": axes
      })
      np.testing.assert_almost_equal(output["S"], x[0:2, 0:2, 0:2])

    # test case 2 with negative, out-of-bound and default inputs
    axes = [0, 2]
    starts = [0, -7]
    ends = [-8, 20]
    steps = [1, 1]

    if legacy_opset_pre_ver(10):
      node_def = helper.make_node("Slice", ["X"], ["S"],
                                  axes=axes,
                                  starts=starts,
                                  ends=ends)
      graph_def = helper.make_graph(
          [node_def],
          name="test_unknown_shape",
          inputs=[
              helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                            [None, None, None])
          ],
          outputs=[
              helper.make_tensor_value_info("S", TensorProto.FLOAT,
                                            [None, None, None])
          ])
    else:
      node_def = helper.make_node("Slice",
                                  ["X", "starts", "ends", "axes", "steps"],
                                  ["S"])
      graph_def = helper.make_graph(
          [node_def],
          name="test_unknown_shape",
          inputs=[
              helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                            [None, None, None]),
              helper.make_tensor_value_info("starts", TensorProto.INT32,
                                            [None]),
              helper.make_tensor_value_info("ends", TensorProto.INT32, [None]),
              helper.make_tensor_value_info("axes", TensorProto.INT32, [None]),
              helper.make_tensor_value_info("steps", TensorProto.INT32, [None]),
          ],
          outputs=[
              helper.make_tensor_value_info("S", TensorProto.FLOAT,
                                            [None, None, None])
          ])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    if legacy_opset_pre_ver(10):
      x = self._get_rnd_float32(shape=[1000]).reshape([10, 10, 10])
      output = tf_rep.run({"X": x})
      np.testing.assert_almost_equal(output["S"], x[0:-8, :, -7:20])
    else:
      x = self._get_rnd_float32(shape=[1000]).reshape([10, 10, 10])
      output = tf_rep.run({
          "X": x,
          "starts": starts,
          "ends": ends,
          "axes": axes,
          "steps": steps
      })
      np.testing.assert_almost_equal(output["S"], x[0:-8, :, -7:20])

    # test case 3 with non-default steps
    axes = [0, 1, 2]
    starts = [0, 0, 0]
    ends = [2, 2, 2]
    steps = [2, -2, -1]

    if not legacy_opset_pre_ver(10):
      x = self._get_rnd_float32(shape=[1000]).reshape([10, 10, 10])
      output = tf_rep.run({
          "X": x,
          "starts": starts,
          "ends": ends,
          "axes": axes,
          "steps": steps
      })
      np.testing.assert_almost_equal(output["S"], x[0:2:2, 0:2:-2, 0:2:-1])


if __name__ == '__main__':
  unittest.main()
