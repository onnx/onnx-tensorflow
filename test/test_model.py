from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import tensorflow as tf
import onnx
from onnx_tf.backend import run_node, prepare
from onnx import helper
from onnx.onnx_pb2 import TensorProto

class TestModel(unittest.TestCase):
  """ Tests for models
  """
  def test_relu_node_inplace(self):
    X = np.random.randn(3, 2).astype(np.float32)
    Y_ref = np.clip(X, 0, np.inf)

    node_def = helper.make_node(
      "Relu", ["X"], ["X"])

    graph_def = helper.make_graph(
      [node_def],
      name="test",
      inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 2])],
      outputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 2])])
    tf_rep = prepare(helper.make_model(graph_def))
    output = tf_rep.run({"X": X})
    np.testing.assert_almost_equal(output.X, Y_ref)

  def test_initializer(self):
    X = np.array([[1, 2], [3, 4]]).astype(np.float32)
    Y = np.array([[1, 2], [3, 4]]).astype(np.float32)
    weight = np.array([[1, 0], [0, 1]])
    graph_def = helper.make_graph(
        [helper.make_node("Add", ["X", "Y"], ["Z0"]),
         helper.make_node("Cast", ["Z0"], ["Z"], to="float"),
         helper.make_node("Mul", ["Z", "weight"], ["W"]),
         helper.make_node("Tanh", ["W"], ["W"]),
         helper.make_node("Sigmoid", ["W"], ["W"])],
        name="test_initializer",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 2)),
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 2)),
            helper.make_tensor_value_info("weight", TensorProto.FLOAT, (2, 2)),
        ],
        outputs=[
            helper.make_tensor_value_info("W", TensorProto.FLOAT, (2, 2))
        ],
        initializer=[helper.make_tensor("weight",
                                 TensorProto.FLOAT,
                                 [2, 2],
                                 weight.flatten().astype(float))]
    )

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    W_ref = sigmoid(np.tanh((X + Y) * weight))
    tf_rep = prepare(helper.make_model(graph_def))
    output = tf_rep.run({"X": X, "Y": Y})
    np.testing.assert_almost_equal(output["W"], W_ref)

if __name__ == '__main__':
  unittest.main()
