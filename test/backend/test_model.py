from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import numpy as np
import onnx
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

  def test_sequence_ops(self):
    # test SequenceConstruct and SequenceAt
    a = np.random.randn(2, 1, 2).astype(np.float32)
    b = np.random.randn(1, 1, 2).astype(np.float32)
    c = np.random.randn(3, 1, 2).astype(np.float32)
    seq_construct_node = helper.make_node('SequenceConstruct', ['a', 'b', 'c'], ['S'])
    seq_at_node = helper.make_node('SequenceAt', ['S','at'], ['Y'])
    out_value_info = helper.make_tensor_value_info('Y',onnx.TensorProto.FLOAT,[None])
    a_value_info =  helper.make_tensor_value_info('a',onnx.TensorProto.FLOAT,[2, 1, 2])
    b_value_info =  helper.make_tensor_value_info('b',onnx.TensorProto.FLOAT,[1, 1, 2])
    c_value_info =  helper.make_tensor_value_info('c',onnx.TensorProto.FLOAT,[3, 1, 2])
    at_value_info =  helper.make_tensor_value_info('at',onnx.TensorProto.INT32,[])

    graph = helper.make_graph([seq_construct_node, seq_at_node],
            name='seq_construct_at_test',
            inputs=[a_value_info, b_value_info, c_value_info, at_value_info],
            outputs=[out_value_info])
    model = helper.make_model(graph, producer_name='backend-test')
    tf_rep = prepare(model)
    output = tf_rep.run({'a':a, 'b':b, 'c':c, 'at':0})
    np.testing.assert_almost_equal(output["Y"], a)
    output = tf_rep.run({'a':a, 'b':b, 'c':c, 'at':-2})
    np.testing.assert_almost_equal(output["Y"], b)
    output = tf_rep.run({'a':a, 'b':b, 'c':c, 'at':2})
    np.testing.assert_almost_equal(output["Y"], c)

    # test SequenceEmpty, SequenceInsert, and SequenceAt
    p = np.int32(0)
    seq_empty_node = helper.make_node('SequenceEmpty', [], ['S'])
    seq_insert_node1 = helper.make_node('SequenceInsert', ['S','a'], ['S1'])
    seq_insert_node2 = helper.make_node('SequenceInsert', ['S1','b'], ['S2'])
    seq_insert_node3 = helper.make_node('SequenceInsert', ['S2','c','p'], ['S3'])
    seq_at_node = helper.make_node('SequenceAt', ['S3','at'], ['Y'])

    p_value_info = helper.make_tensor_value_info('p',onnx.TensorProto.INT32,[])

    graph = helper.make_graph([seq_empty_node, seq_insert_node1, seq_insert_node2, seq_insert_node3, seq_at_node],
            name='seq_empty_insert_at_test',
            inputs=[a_value_info, b_value_info, c_value_info, p_value_info, at_value_info],
            outputs=[out_value_info])
    model = helper.make_model(graph, producer_name='backend-test')
    tf_rep = prepare(model)
    output = tf_rep.run({'a':a, 'b':b, 'c':c, 'p':p, 'at':0})
    np.testing.assert_almost_equal(output["Y"], c)

    # test SequenceConstruct, SequenceErase, and SequenceLength
    seq_construct_node = helper.make_node('SequenceConstruct', ['a', 'b', 'c'], ['S'])
    seq_erase_node = helper.make_node('SequenceErase', ['S','p'], ['S1'])
    seq_length_node = helper.make_node('SequenceLength', ['S1'], ['Y'])

    graph = helper.make_graph([seq_construct_node, seq_erase_node, seq_length_node],
            name='seq_construct_erase_length_test',
            inputs=[a_value_info, b_value_info, c_value_info, p_value_info],
            outputs=[out_value_info])
    model = helper.make_model(graph, producer_name='backend-test')
    tf_rep = prepare(model)
    output = tf_rep.run({'a':a, 'b':b, 'c':c, 'p':p})
    np.testing.assert_almost_equal(output["Y"], 2)

    # test SequenceConstruct and SequenceErase
    seq_construct_node = helper.make_node('SequenceConstruct', ['a', 'b', 'c'], ['S'])
    seq_erase_node = helper.make_node('SequenceErase', ['S','p'], ['S1'])
    seq_at_node = helper.make_node('SequenceAt', ['S1', 'at'], ['Y'])

    graph = helper.make_graph([seq_construct_node, seq_erase_node, seq_at_node],
            name='seq_construct_erase_test',
            inputs=[a_value_info, b_value_info, c_value_info, p_value_info, at_value_info],
            outputs=[out_value_info])
    model = helper.make_model(graph, producer_name='backend-test')
    tf_rep = prepare(model)
    output = tf_rep.run({'a':a, 'b':b, 'c':c, 'p':p, 'at':0})
    np.testing.assert_almost_equal(output["Y"], b)
    output = tf_rep.run({'a':a, 'b':b, 'c':c, 'p':p, 'at':1})
    np.testing.assert_almost_equal(output["Y"], c)

    # test SequenceConstruct and ConcatFromSequence
    seq_construct_node = helper.make_node('SequenceConstruct', ['a', 'b', 'c'], ['S'])
    concat_from_seq_node = helper.make_node('ConcatFromSequence', ['S'], ['Y'], axis=1)
    a = [[1, 2],[3, 4]]
    b = [[5, 6],[7, 8]]
    c = [[9, 10],[11, 12]]
    a_value_info =  helper.make_tensor_value_info('a',onnx.TensorProto.FLOAT,[2, 2])
    b_value_info =  helper.make_tensor_value_info('b',onnx.TensorProto.FLOAT,[2, 2])
    c_value_info =  helper.make_tensor_value_info('c',onnx.TensorProto.FLOAT,[2, 2])

    graph = helper.make_graph([seq_construct_node, concat_from_seq_node],
            name='seq_construct_concat_test',
            inputs=[a_value_info, b_value_info, c_value_info],
            outputs=[out_value_info])
    model = helper.make_model(graph, producer_name='backend-test')
    tf_rep = prepare(model)
    output = tf_rep.run({'a':a, 'b':b, 'c':c})
    d = np.concatenate((a, b, c), axis=1).astype(np.float32)
    np.testing.assert_almost_equal(output["Y"], d)

    # test SplitToSequence and SequenceAt
    a= np.array([[1,2,3,4,5,6,7],
        [11,12,13,14,15,16,17],
        [21,22,23,24,25,26,27]]).astype(np.float32)
    b = np.int32([2,1])
    seq_split_node = helper.make_node('SplitToSequence', ['a','b'], ['S'])
    seq_at_node = helper.make_node('SequenceAt', ['S','at'], ['Y'])
    a_value_info =  helper.make_tensor_value_info('a',onnx.TensorProto.FLOAT,[3,7])
    b_value_info =  helper.make_tensor_value_info('b',onnx.TensorProto.INT32,[2])
    at_value_info =  helper.make_tensor_value_info('at',onnx.TensorProto.INT32,[])

    graph = helper.make_graph([seq_split_node, seq_at_node],
            name='split_to_seq_test',
            inputs=[a_value_info, b_value_info, at_value_info],
            outputs=[out_value_info])
    model = helper.make_model(graph, producer_name='backend-test')
    tf_rep = prepare(model)
    output = tf_rep.run({'a':a, 'b':b, 'at':1})
    np.testing.assert_almost_equal(output["Y"], np.split(a, [2,3])[1])

    axis=1
    seq_split_node = helper.make_node('SplitToSequence', ['a'], ['S'], axis=axis)
    seq_at_node = helper.make_node('SequenceAt', ['S','at'], ['Y'])
    at_value_info =  helper.make_tensor_value_info('at',onnx.TensorProto.INT32,[])

    graph = helper.make_graph([seq_split_node, seq_at_node],
            name='split_to_seq_test',
            inputs=[a_value_info, at_value_info],
            outputs=[out_value_info])
    model = helper.make_model(graph, producer_name='backend-test')
    tf_rep = prepare(model)
    output = tf_rep.run({'a':a, 'at':0})
    np.testing.assert_almost_equal(output["Y"], np.split(a, 7, axis=1)[0])

    seq_split_node = helper.make_node('SplitToSequence', ['a'], ['S'], keepdims=0)
    seq_at_node = helper.make_node('SequenceAt', ['S','at'], ['Y'])
    at_value_info =  helper.make_tensor_value_info('at',onnx.TensorProto.INT32,[])

    graph = helper.make_graph([seq_split_node, seq_at_node],
            name='split_to_seq_test',
            inputs=[a_value_info, at_value_info],
            outputs=[out_value_info])
    model = helper.make_model(graph, producer_name='backend-test')
    tf_rep = prepare(model)
    output = tf_rep.run({'a':a, 'at':0})
    expected = [np.squeeze(res) for res in np.split(a, 3)]
    np.testing.assert_almost_equal(output["Y"], expected[0])

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

if __name__ == '__main__':
  unittest.main()
