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
    if shape == None:
      return np.float32(output)
    else:
      return output.astype(np.float32)

  def _get_rnd_int(self, low, high=None, shape=None, dtype=np.int32):
    return np.random.randint(low, high, size=shape, dtype=dtype)

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
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                          [None, None, None]),
            helper.make_tensor_value_info("condition", TensorProto.BOOL, [None])
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT,
                                          [None, None, None])
        ])
    x = self._get_rnd_float32(shape=[5, 5, 5])
    cond = np.array([1, 0, 1])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    output = tf_rep.run({"X": x, "condition": cond})
    np.testing.assert_almost_equal(output['Y'], np.compress(cond, x, axis=axis))

  def test_eye_like(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support EyeLike.".format(
          defs.onnx_opset_version()))
    shape = [6, 10]
    off_diagonal_offset = -3
    x = self._get_rnd_int(0, 100, shape=shape)
    y = np.eye(shape[0], shape[1], k=off_diagonal_offset, dtype=np.float32)
    node_def = helper.make_node("EyeLike", ["x"], ["y"],
                                dtype=TensorProto.FLOAT,
                                k=off_diagonal_offset)
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.INT32, [None, None])
        ],
        outputs=[
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, None])
        ])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    output = tf_rep.run({"x": x})
    np.testing.assert_equal(output["y"], y)

  def test_flatten(self):
    shape = [2, 3, 4]
    x = self._get_rnd_float32(shape=shape)
    axis = 1
    node_def = helper.make_node("Flatten", ["X"], ["Y"], axis=axis)
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                          [None, None, None])
        ],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    output = tf_rep.run({"X": x})
    new_shape = (np.prod(shape[0:axis]).astype(int), -1)
    np.testing.assert_almost_equal(output["Y"], np.reshape(x, new_shape))

  def test_gather_nd(self):
    if legacy_opset_pre_ver(11):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support GatherND.".format(
              defs.onnx_opset_version()))
    # valid positive and negative indices for elements
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    indices = np.array([[0, 0], [1, -3]], dtype=np.int64)
    ref_output = np.array([1, 4], dtype=np.int32)
    node_def = helper.make_node("GatherND", ["data", "indices"], ["outputs"])
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.INT32,
                                          [None, None]),
            helper.make_tensor_value_info("indices", TensorProto.INT64,
                                          [None, None])
        ],
        outputs=[
            helper.make_tensor_value_info("outputs", TensorProto.INT32, [None])
        ])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    output = tf_rep.run({"data": data, "indices": indices})
    np.testing.assert_almost_equal(output["outputs"], ref_output)

  def test_is_inf(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest("ONNX version {} doesn't support IsInf.".format(
          defs.onnx_opset_version()))
    inp = np.array([-1.2, np.nan, np.inf, 2.8, np.NINF, np.inf],
                   dtype=np.float32)
    expected_output = np.isinf(inp)
    node_def = helper.make_node("IsInf", ["X"], ["Y"])
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [None]),
        ],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.BOOL, [None])])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    output = tf_rep.run({"X": inp})
    np.testing.assert_equal(output["Y"], expected_output)

  def test_scatter_nd(self):
    if legacy_opset_pre_ver(11):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support ScatterND.".format(
              defs.onnx_opset_version()))
    # valid positive and negative indices for slices
    data = np.reshape(np.arange(1, 25, dtype=np.float32), [2, 3, 4])
    indices = np.array([[-1]], dtype=np.int64)
    updates = np.array([[[43, 44, 45, 46], [47, 48, 49, 50], [51, 52, 53, 54]]],
                       dtype=np.float32)
    ref_output = np.array(
        [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
         [[43, 44, 45, 46], [47, 48, 49, 50], [51, 52, 53, 54]]],
        dtype=np.float32)
    node_def = helper.make_node("ScatterND", ["data", "indices", "updates"],
                                ["outputs"])
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT,
                                          [None, None, None]),
            helper.make_tensor_value_info("indices", TensorProto.INT64,
                                          [None, None]),
            helper.make_tensor_value_info("updates", TensorProto.FLOAT,
                                          [None, None, None])
        ],
        outputs=[
            helper.make_tensor_value_info("outputs", TensorProto.FLOAT,
                                          [None, None, None])
        ])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    output = tf_rep.run({"data": data, "indices": indices, "updates": updates})
    np.testing.assert_almost_equal(output["outputs"], ref_output)

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

    test_output = py_pool(x, kernel_shape=kernel_shape, strides=strides,
                          dilations=dilations, padding=pads,
                          ceil_mode=ceil_mode, pooling_type="MAX",
                          include_indices=False)

    node_def = helper.make_node(
            op_type="MaxPool",
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

  def test_average_pool_2d(self):
    kernel_shape = [1, 2]
    strides = [1, 2]

    input_shape = [10, 10, 4, 4]
    x = self._get_rnd_float32(shape=input_shape)

    test_output = py_pool(x, kernel_shape=kernel_shape, strides=strides,
                          pooling_type="AVG", include_indices=False)

    node_def = helper.make_node(
            op_type="AveragePool",
            inputs=["X"],
            outputs=["Y"],
            kernel_shape=kernel_shape,
            strides=strides)

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

  def test_max_unpool(self):
    input_shape = [10, 3, 24, 24]
    x = self._get_rnd_float32(shape=input_shape)

    kernel_shape = [2, 2]
    strides = [2, 2]

    maxpool_node_def = helper.make_node(
            op_type="MaxPool",
            inputs=["X"],
            outputs=["Pool", "Indices"],
            kernel_shape=kernel_shape,
            strides=strides)

    maxunpool_node_def = helper.make_node(
        "MaxUnpool", ["Pool", "Indices"], ["Y"],
        kernel_shape=kernel_shape,
        strides=strides)

    graph_def = helper.make_graph(
        [maxpool_node_def,maxunpool_node_def],
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
    output_unpool = tf_rep.run({"X": x})

    test_output = np.zeros(input_shape)
    for i1 in range(0, input_shape[0]):
      for i2 in range(0, input_shape[1]):
        for i3 in range(0, input_shape[2], 2):
          for i4 in range(0, input_shape[3], 2):
            max_val = float('-inf')
            for j1 in range(i3,i3+2):
              for j2 in range(i4,i4+2):
                if x[i1][i2][j1][j2] > max_val:
                  max_val = x[i1][i2][j1][j2]
                  max_ind = (j1, j2)
            j1, j2 = max_ind
            test_output[i1][i2][j1][j2] = max_val
    np.testing.assert_almost_equal(output_unpool["Y"], test_output)

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
      node_def = helper.make_node("Slice",
                                  ["X", "starts", "ends", "axes"],
                                  ["S"])
      graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
          helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                        [None, None, None]),
          helper.make_tensor_value_info("starts", TensorProto.INT32,
                                        [None]),
          helper.make_tensor_value_info("ends", TensorProto.INT32,
                                        [None]),
          helper.make_tensor_value_info("axes", TensorProto.INT32,
                                        [None]),
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
      output = tf_rep.run({"X": x, "starts": starts, "ends": ends, "axes": axes})
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
          helper.make_tensor_value_info("ends", TensorProto.INT32,
                                        [None]),
          helper.make_tensor_value_info("axes", TensorProto.INT32,
                                        [None]),
          helper.make_tensor_value_info("steps", TensorProto.INT32,
                                        [None]),
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
      output = tf_rep.run({"X": x, "starts": starts, "ends": ends, "axes": axes, "steps": steps})
      np.testing.assert_almost_equal(output["S"], x[0:-8, :, -7:20])

    # test case 3 with non-default steps
    axes = [0, 1, 2]
    starts = [0, 0, 0]
    ends = [2, 2, 2]
    steps = [2, -2, -1]

    if not legacy_opset_pre_ver(10):
      x = self._get_rnd_float32(shape=[1000]).reshape([10, 10, 10])
      output = tf_rep.run({"X": x, "starts": starts, "ends": ends, "axes": axes, "steps": steps})
      np.testing.assert_almost_equal(output["S"], x[0:2:2, 0:2:-2, 0:2:-1])

if __name__ == '__main__':
  unittest.main()
