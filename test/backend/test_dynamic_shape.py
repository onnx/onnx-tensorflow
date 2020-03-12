from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import unittest

from onnx_tf.backend import onnx_graph_to_tensorflow_rep
from onnx_tf.common.legacy import legacy_opset_pre_ver
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
