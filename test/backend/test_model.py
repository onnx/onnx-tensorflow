from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import numpy as np
from onnx_tf.backend import prepare
from onnx import helper
from onnx import TensorProto

from onnx_tf.common.legacy import legacy_onnx_pre_ver


class TestModel(unittest.TestCase):
  """ Tests for models
  """

  def _get_rnd(self, shape, low=-1.0, high=1.0):
    return np.random.uniform(low, high, np.prod(shape)) \
                      .reshape(shape) \
                      .astype(np.float32)

  def test_relu_node_inplace(self):
    X = np.random.randn(3, 2).astype(np.float32)
    Y_ref = np.clip(X, 0, np.inf)

    node_def = helper.make_node("Relu", ["X"], ["X1"])

    graph_def = helper.make_graph(
        [node_def],
        name="test",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 2])],
        outputs=[
            helper.make_tensor_value_info("X1", TensorProto.FLOAT, [3, 2])
        ])
    tf_rep = prepare(helper.make_model(graph_def))
    output = tf_rep.run({"X": X})
    np.testing.assert_almost_equal(output.X1, Y_ref)

  def test_initializer(self):
    if legacy_onnx_pre_ver(1, 2):
      raise unittest.SkipTest(
          "The current version of ONNX does not record correctly the opset of Cast."
      )
    X = np.array([[1, 2], [3, 4]]).astype(np.float32)
    Y = np.array([[1, 2], [3, 4]]).astype(np.float32)
    weight = np.array([[1, 0], [0, 1]])
    graph_def = helper.make_graph(
        [
            helper.make_node("Add", ["X", "Y"], ["Z0"]),
            helper.make_node("Cast", ["Z0"], ["Z"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["Z", "weight"], ["W"]),
            helper.make_node("Tanh", ["W"], ["W1"]),
            helper.make_node("Sigmoid", ["W1"], ["W2"])
        ],
        name="test_initializer",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 2)),
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 2)),
            helper.make_tensor_value_info("weight", TensorProto.FLOAT, (2, 2)),
        ],
        outputs=[
            helper.make_tensor_value_info("W2", TensorProto.FLOAT, (2, 2))
        ],
        initializer=[
            helper.make_tensor("weight", TensorProto.FLOAT, [2, 2],
                               weight.flatten().astype(float))
        ])

    def sigmoid(x):
      return 1 / (1 + np.exp(-x))

    W_ref = sigmoid(np.tanh((X + Y) * weight))
    tf_rep = prepare(helper.make_model(graph_def))
    output = tf_rep.run({"X": X, "Y": Y})
    np.testing.assert_almost_equal(output["W2"], W_ref)

  def test_max_unpool(self):
    input_shape = [10,10,4,4]
    x = self._get_rnd(input_shape)

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_shape)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, input_shape)

    maxpool_node_def = helper.make_node(
        "MaxPool", ["X"], ["Pool", "Indices"],
        kernel_shape=[2, 2],
        strides=[2, 2])

    maxunpool_node_def = helper.make_node(
        "MaxUnpool", ["Pool", "Indices"], ["Y"],
        kernel_shape=[2, 2],
        strides=[2, 2])

    graph_def = helper.make_graph(
        [maxpool_node_def,maxunpool_node_def],
        "MaxUnpool-model",
        [X],
        [Y],
    )

    """ Maxpool op version 10 is not implemented yet and that is why we use a workaround
        to force the onnx ops set version to 9
    """
    version = helper.make_operatorsetid("",9)
    model_def = helper.make_model(graph_def,
                                  opset_imports=[version])
    tf_rep = prepare(model_def)  # run the loaded model
    output_unpool = tf_rep.run(x)

    """ This code is simpler way to test maxunpool but fails because
        maxpool op version 10 is not supported yet. Once maxpool op version 10
        is supported this code can be moved to more simple test_node

    node_def = helper.make_node(
        "MaxPool", ["X"], ["Pool", "Indices"],
        kernel_shape=[2, 2],
        strides=[2, 2])
    output_pool = run_node(node_def, [x])

    node_def = helper.make_node(
        "MaxUnpool", ["Pool", "Indices"], ["Y"],
        kernel_shape=[2, 2],
        strides=[2, 2])
    output_unpool = run_node(node_def, [output_pool["Pool"], output_pool["Indices"]])
    """

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

if __name__ == '__main__':
  unittest.main()
