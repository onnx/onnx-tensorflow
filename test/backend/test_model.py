from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import shutil

import tensorflow as tf
import numpy as np
import onnx
from onnx_tf.backend import prepare
from onnx import defs
from onnx import helper
from onnx import TensorProto
from onnx.backend.test.case.node.lstm import LSTM_Helper
from onnx.backend.test.case.node.gru import GRU_Helper
from onnx.backend.test.case.node.rnn import RNN_Helper

from onnx_tf.common.legacy import legacy_onnx_pre_ver
from onnx_tf.common.legacy import legacy_opset_pre_ver


class TestModel(unittest.TestCase):
  """ Tests for models
  """

  def _get_rnd(self, shape, low=-1.0, high=1.0):
    return np.random.uniform(low, high, np.prod(shape)) \
                      .reshape(shape) \
                      .astype(np.float32)

  def test_lstm_savedmodel(self):
    input_size = 4
    hidden_size = 3
    weight_scale = 0.1
    number_of_gates = 4

    node_def = helper.make_node(
        'LSTM',
        inputs=['X', 'W', 'R', 'B', 'sequence_lens', 'initial_h'],
        outputs=['Y', 'Y_h', 'Y_c'],
        hidden_size=hidden_size)
    graph_def = helper.make_graph(
        [node_def],
        name="lstm_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 4]),
            helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 12, 4]),
            helper.make_tensor_value_info("R", TensorProto.FLOAT, [1, 12, 3]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 24]),
            helper.make_tensor_value_info("sequence_lens", TensorProto.INT32,
                                          [2]),
            helper.make_tensor_value_info("initial_h", TensorProto.FLOAT,
                                          [1, 2, 3])
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 2, 3]),
            helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, [1, 2, 3]),
            helper.make_tensor_value_info("Y_c", TensorProto.FLOAT, [1, 2, 3])
        ])

    # Initializing Inputs
    X = np.array([[[1., 2., 3., 4.], [5., 6., 7., 8.]]]).astype(np.float32)
    W = weight_scale * np.ones(
        (1, number_of_gates * hidden_size, input_size)).astype(np.float32)
    R = weight_scale * np.ones(
        (1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)
    B = np.zeros((1, 2 * number_of_gates * hidden_size)).astype(np.float32)
    seq_lens = np.repeat(X.shape[0], X.shape[1]).astype(np.int32)
    init_h = weight_scale * np.ones(
        (1, X.shape[1], hidden_size)).astype(np.float32)

    # prepare the ONNX model and save it as a Tensorflow SavedModel
    tf_rep = prepare(helper.make_model(graph_def))
    tf_rep.run({
        "X": X,
        "W": W,
        "R": R,
        "B": B,
        "sequence_lens": seq_lens,
        "initial_h": init_h
    })
    model_path = "lstm_savedmodel"
    tf_rep.export_graph(model_path)

    # use the ONNX reference implementation to get expected output
    lstm = LSTM_Helper(X=X, W=W, R=R, B=B, initial_h=init_h)
    _, Y_ref = lstm.step()

    # load the SavedModel from file system
    m = tf.saved_model.load(model_path)

    # run the model
    tf_output = m(X=X, W=W, R=R, B=B, sequence_lens=seq_lens, initial_h=init_h)

    np.testing.assert_almost_equal(tf_output["Y_h"], Y_ref)

    # clean up saved model folder
    shutil.rmtree(model_path)

  def test_gru_savedmodel(self):
    input_size = 3
    hidden_size = 3
    weight_scale = 0.1
    custom_bias = 0.1
    number_of_gates = 3

    node_def = helper.make_node('GRU',
                                inputs=['X', 'W', 'R', 'B'],
                                outputs=['Y', 'Y_h'],
                                hidden_size=hidden_size)
    graph_def = helper.make_graph(
        [node_def],
        name="gru_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 3]),
            helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 9, 3]),
            helper.make_tensor_value_info("R", TensorProto.FLOAT, [1, 9, 3]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 18])
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 3, 3]),
            helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, [1, 3, 3])
        ])

    # initializing Inputs
    X = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8.,
                                                9.]]]).astype(np.float32)
    W = weight_scale * np.ones(
        (1, number_of_gates * hidden_size, input_size)).astype(np.float32)
    R = weight_scale * np.ones(
        (1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)
    W_B = custom_bias * np.ones(
        (1, number_of_gates * hidden_size)).astype(np.float32)
    R_B = np.zeros((1, number_of_gates * hidden_size)).astype(np.float32)
    B = np.concatenate((W_B, R_B), axis=1)

    # prepare the ONNX model and save it as a Tensorflow SavedModel
    tf_rep = prepare(helper.make_model(graph_def))
    tf_rep.run({"X": X, "W": W, "R": R, "B": B})
    model_path = "gru_savedmodel"
    tf_rep.export_graph(model_path)

    # use the ONNX reference implementation to get the expected output
    gru = GRU_Helper(X=X, W=W, R=R, B=B)
    _, Y_ref = gru.step()

    # load the savedmodel from file system
    m = tf.saved_model.load(model_path)

    # run the model
    tf_output = m(X=X, W=W, R=R, B=B)

    np.testing.assert_almost_equal(tf_output["Y_h"], Y_ref)

    # clean up saved model folder
    shutil.rmtree(model_path)

  def test_rnn_savedmodel(self):
    input_size = 2
    hidden_size = 4
    weight_scale = 0.1

    node_def = helper.make_node('RNN',
                                inputs=['X', 'W', 'R'],
                                outputs=['Y', 'Y_h'],
                                hidden_size=hidden_size)
    graph_def = helper.make_graph(
        [node_def],
        name="rnn_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 2]),
            helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 4, 2]),
            helper.make_tensor_value_info("R", TensorProto.FLOAT, [1, 4, 4])
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 2, 4]),
            helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, [1, 2, 4])
        ])

    # initializing Inputs
    X = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)
    W = weight_scale * np.ones((1, hidden_size, input_size)).astype(np.float32)
    R = weight_scale * np.ones((1, hidden_size, hidden_size)).astype(np.float32)

    # prepare the ONNX model and save it as a Tensorflow SavedModel
    tf_rep = prepare(helper.make_model(graph_def))
    tf_rep.run({"X": X, "W": W, "R": R})
    model_path = "rnn_savedmodel"
    tf_rep.export_graph(model_path)

    # use the ONNX reference implementation to get the expected output
    rnn = RNN_Helper(X=X, W=W, R=R)
    _, Y_ref = rnn.step()

    # load the savedmodel from file system
    m = tf.saved_model.load(model_path)

    # run the model
    tf_output = m(X=X, W=W, R=R)

    np.testing.assert_almost_equal(tf_output["Y_h"], Y_ref)

    # clean up saved model folder
    shutil.rmtree(model_path)

  def test_auto_cast(self):
    node_def = helper.make_node("Equal", ["a", "b"], ["Y"])
    graph_def = helper.make_graph(
        [node_def],
        name="test_auto_cast",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.UINT64,
                                          [None, None]),
            helper.make_tensor_value_info("b", TensorProto.UINT64, [None, None])
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.BOOL, [None, None])
        ])
    tf_rep = prepare(helper.make_model(graph_def), auto_cast=True)

    # random inputs with shape [5, 5]
    a = np.random.randint(low=0, high=10, size=(5, 5)).astype(np.uint64)
    b = np.random.randint(low=0, high=10, size=(5, 5)).astype(np.uint64)
    Y_ref = np.equal(a, b)

    # check the output from converter API against numpy's output
    output = tf_rep.run({"a": a, "b": b})
    np.testing.assert_almost_equal(output.Y, Y_ref)

  def test_add_module(self):
    node_def = helper.make_node("Add", ["a", "b"], ["Y"])
    graph_def = helper.make_graph(
        [node_def],
        name="test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [None, None]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [None, None])
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        ])
    tf_rep = prepare(helper.make_model(graph_def))

    model_path = "add_savedmodel"
    tf_rep.export_graph(model_path)

    # loead the model from disk
    m = tf.saved_model.load(model_path)

    # first input with shape [3, 2]
    a = np.random.randn(3, 2).astype(np.float32)
    b = np.random.randn(3, 2).astype(np.float32)
    Y_ref = np.add(a, b)

    # check the output from converter API
    output = tf_rep.run({"a": a, "b": b})
    np.testing.assert_almost_equal(output.Y, Y_ref)

    # check the output from using saved model
    tf_output = m(a=a, b=b)
    np.testing.assert_almost_equal(tf_output["Y"], Y_ref)

    # change input shape to [2, 2]
    a = np.random.randn(2, 2).astype(np.float32)
    b = np.random.randn(2, 2).astype(np.float32)
    Y_ref = np.add(a, b)

    tf_output = m(a=a, b=b)
    np.testing.assert_almost_equal(tf_output["Y"], Y_ref)

    # clean up saved model folder
    shutil.rmtree(model_path)

  def test_argmax_node_bfloat(self):
    X = np.random.randn(2, 8).astype(np.float32)
    Y_ref = np.argmax(X, axis=0)

    graph_def = helper.make_graph(
        [
            helper.make_node("Cast", ["X0"], ["X1"], to=TensorProto.BFLOAT16),
            helper.make_node("ArgMax", ["X1"], ["X2"], axis=0, keepdims=0)
        ],
        name="test",
        inputs=[helper.make_tensor_value_info("X0", TensorProto.FLOAT, [2, 8])],
        outputs=[
            helper.make_tensor_value_info("X2", TensorProto.BFLOAT16, [2, 8])
        ])
    tf_rep = prepare(helper.make_model(graph_def))
    output = tf_rep.run({"X0": X})
    np.testing.assert_almost_equal(output.X2, Y_ref)

  def test_argmin_node_bfloat(self):
    X = np.random.randn(2, 8).astype(np.float32)
    Y_ref = np.argmin(X, axis=0)

    graph_def = helper.make_graph(
        [
            helper.make_node("Cast", ["X0"], ["X1"], to=TensorProto.BFLOAT16),
            helper.make_node("ArgMin", ["X1"], ["X2"], axis=0, keepdims=0)
        ],
        name="test",
        inputs=[helper.make_tensor_value_info("X0", TensorProto.FLOAT, [2, 8])],
        outputs=[
            helper.make_tensor_value_info("X2", TensorProto.BFLOAT16, [2, 8])
        ])
    tf_rep = prepare(helper.make_model(graph_def))
    output = tf_rep.run({"X0": X})
    np.testing.assert_almost_equal(output.X2, Y_ref)

  def test_if_with_sequence(self):
    if legacy_opset_pre_ver(14):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support helper.make_tensor_sequence_value_info."
          .format(defs.onnx_opset_version()))

    # S = [a]
    # if cond is True
    #   S = [a,b]
    # else
    #   S = [a,c]
    a = np.random.randn(2, 1, 2).astype(np.float32)
    b = np.random.randn(1, 1, 2).astype(np.float32)
    c = np.random.randn(3, 1, 2).astype(np.float32)
    seq_construct_node = helper.make_node('SequenceConstruct', ['a'], ['S'])
    seq_insert_node1 = helper.make_node('SequenceInsert', ['S', 'b'], ['Sb'])
    seq_insert_node2 = helper.make_node('SequenceInsert', ['S', 'c'], ['Sc'])

    a_in = helper.make_tensor_value_info('a', onnx.TensorProto.FLOAT, [2, 1, 2])
    b_in = helper.make_tensor_value_info('b', onnx.TensorProto.FLOAT, [1, 1, 2])
    c_in = helper.make_tensor_value_info('c', onnx.TensorProto.FLOAT, [3, 1, 2])
    cond_in = helper.make_tensor_value_info('cond', TensorProto.BOOL, [])
    s_in = helper.make_sequence_value_info('S', TensorProto.FLOAT,
                                                  [None, None, None, None])

    sb_out = helper.make_sequence_value_info('Sb', TensorProto.FLOAT,
                                                    [None, None, None, None])
    sc_out = helper.make_sequence_value_info('Sc', TensorProto.FLOAT,
                                                    [None, None, None, None])
    s_final_out = helper.make_sequence_value_info(
        'S_final', TensorProto.FLOAT, [None, None, None, None])

    then_graph = helper.make_graph(nodes=[seq_insert_node1],
                                   name="then_graph",
                                   inputs=[s_in, b_in],
                                   outputs=[sb_out])
    else_graph = helper.make_graph(nodes=[seq_insert_node2],
                                   name="else_graph",
                                   inputs=[s_in, c_in],
                                   outputs=[sc_out])
    if_node = helper.make_node('If', ['cond'], ['S_final'],
                               then_branch=then_graph,
                               else_branch=else_graph)

    graph_def = helper.make_graph(nodes=[seq_construct_node, if_node],
                                  name='test_if',
                                  inputs=[a_in, b_in, c_in, cond_in],
                                  outputs=[s_final_out])
    tf_rep = prepare(helper.make_model(graph_def))
    output = tf_rep.run({
        'a': a,
        'b': b,
        'c': c,
        'cond': np.array(True, dtype=np.bool)
    })
    np.testing.assert_almost_equal(output['S_final'].values[:2], a)
    np.testing.assert_almost_equal(output['S_final'].values[2:], b)
    output = tf_rep.run({
        'a': a,
        'b': b,
        'c': c,
        'cond': np.array(False, dtype=np.bool)
    })
    np.testing.assert_almost_equal(output['S_final'].values[:2], a)
    np.testing.assert_almost_equal(output['S_final'].values[2:], c)

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

  def test_loop_with_sequence(self):
    if legacy_opset_pre_ver(14):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support helper.make_tensor_sequence_value_info."
          .format(defs.onnx_opset_version()))

    # construct sequence S with tensor a in it
    # insert tensor b into sequence S for 3 time
    a = np.random.randn(2, 1, 2).astype(np.float32)
    b = np.random.randn(1, 1, 2).astype(np.float32)
    M = np.array(3, dtype=np.int64)
    cond = np.array(True, dtype=np.bool)
    seq_construct_node = helper.make_node('SequenceConstruct', ['a'], ['S'])
    seq_insert_node = helper.make_node('SequenceInsert', ['S', 'b'],
                                       ['Updated_S'])

    a_in = helper.make_tensor_value_info('a', onnx.TensorProto.FLOAT, [2, 1, 2])
    b_in = helper.make_tensor_value_info('b', onnx.TensorProto.FLOAT, [1, 1, 2])
    M_in = helper.make_tensor_value_info('M', TensorProto.INT64, [])
    cond_init_in = helper.make_tensor_value_info('cond_init', TensorProto.BOOL,
                                                 [])
    iter_count_in = helper.make_tensor_value_info('iter_count',
                                                  TensorProto.INT64, [])
    cond_in = helper.make_tensor_value_info('cond', TensorProto.BOOL, [])
    s_in = helper.make_sequence_value_info('S', TensorProto.FLOAT,
                                                  [None, None, None, None])

    cond_out = helper.make_tensor_value_info('cond', TensorProto.BOOL, [])
    s_out = helper.make_sequence_value_info('Updated_S',
                                                   TensorProto.FLOAT,
                                                   [None, None, None, None])
    s_final_out = helper.make_sequence_value_info(
        'S_final', TensorProto.FLOAT, [None, None, None, None])

    body_graph = helper.make_graph(nodes=[seq_insert_node],
                                   name="for_loop_graph",
                                   inputs=[iter_count_in, cond_in, s_in],
                                   outputs=[cond_out, s_out])
    loop_node = helper.make_node('Loop', ['M', '', 'S'], ['S_final'],
                                 body=body_graph)

    graph_def = helper.make_graph(nodes=[seq_construct_node, loop_node],
                                  name='test_loop',
                                  inputs=[a_in, b_in, M_in, cond_init_in],
                                  outputs=[s_final_out])
    tf_rep = prepare(helper.make_model(graph_def))
    output = tf_rep.run({'a': a, 'b': b, 'M': M, 'cond_init': cond})
    np.testing.assert_almost_equal(output['S_final'].values[:2], a)
    np.testing.assert_almost_equal(output['S_final'].values[2:3], b)
    np.testing.assert_almost_equal(output['S_final'].values[3:4], b)
    np.testing.assert_almost_equal(output['S_final'].values[4:5], b)

  def test_pow_bfloat16(self):
    X1 = np.array([1, 2, 3]).astype(np.float32)
    X2 = np.array([2, 3, 4]).astype(np.float32)
    Y_ref = np.power(X1, X2).astype(X1.dtype)

    graph_def = helper.make_graph(
        [
            helper.make_node("Cast", ["X1"], ["C1"], to=TensorProto.BFLOAT16),
            helper.make_node("Pow", ["C1", "X2"], ["Y"])
        ],
        name="test",
        inputs=[
            helper.make_tensor_value_info("X1", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("X2", TensorProto.FLOAT, [3])
        ],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [3])])
    tf_rep = prepare(helper.make_model(graph_def))
    output = tf_rep.run({"X1": X1, "X2": X2})
    np.testing.assert_almost_equal(output.Y, Y_ref)

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

  def test_sequence_ops(self):
    # test SequenceConstruct and SequenceAt
    a = np.random.randn(2, 1, 2).astype(np.float32)
    b = np.random.randn(1, 1, 2).astype(np.float32)
    c = np.random.randn(3, 1, 2).astype(np.float32)
    seq_construct_node = helper.make_node('SequenceConstruct', ['a', 'b', 'c'],
                                          ['S'])
    seq_at_node = helper.make_node('SequenceAt', ['S', 'at'], ['Y'])
    out_value_info = helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT,
                                                   [None])
    a_value_info = helper.make_tensor_value_info('a', onnx.TensorProto.FLOAT,
                                                 [2, 1, 2])
    b_value_info = helper.make_tensor_value_info('b', onnx.TensorProto.FLOAT,
                                                 [1, 1, 2])
    c_value_info = helper.make_tensor_value_info('c', onnx.TensorProto.FLOAT,
                                                 [3, 1, 2])
    at_value_info = helper.make_tensor_value_info('at', onnx.TensorProto.INT32,
                                                  [])

    graph = helper.make_graph(
        [seq_construct_node, seq_at_node],
        name='seq_construct_at_test',
        inputs=[a_value_info, b_value_info, c_value_info, at_value_info],
        outputs=[out_value_info])
    model = helper.make_model(graph, producer_name='backend-test')
    tf_rep = prepare(model)
    output = tf_rep.run({'a': a, 'b': b, 'c': c, 'at': 0})
    np.testing.assert_almost_equal(output["Y"], a)
    output = tf_rep.run({'a': a, 'b': b, 'c': c, 'at': -2})
    np.testing.assert_almost_equal(output["Y"], b)
    output = tf_rep.run({'a': a, 'b': b, 'c': c, 'at': 2})
    np.testing.assert_almost_equal(output["Y"], c)

    # test SequenceEmpty, SequenceInsert, and SequenceAt
    p = np.int32(0)
    seq_empty_node = helper.make_node('SequenceEmpty', [], ['S'])
    seq_insert_node1 = helper.make_node('SequenceInsert', ['S', 'a'], ['S1'])
    seq_insert_node2 = helper.make_node('SequenceInsert', ['S1', 'b'], ['S2'])
    seq_insert_node3 = helper.make_node('SequenceInsert', ['S2', 'c', 'p'],
                                        ['S3'])
    seq_at_node = helper.make_node('SequenceAt', ['S3', 'at'], ['Y'])

    p_value_info = helper.make_tensor_value_info('p', onnx.TensorProto.INT32,
                                                 [])

    graph = helper.make_graph([
        seq_empty_node, seq_insert_node1, seq_insert_node2, seq_insert_node3,
        seq_at_node
    ],
                              name='seq_empty_insert_at_test',
                              inputs=[
                                  a_value_info, b_value_info, c_value_info,
                                  p_value_info, at_value_info
                              ],
                              outputs=[out_value_info])
    model = helper.make_model(graph, producer_name='backend-test')
    tf_rep = prepare(model)
    output = tf_rep.run({'a': a, 'b': b, 'c': c, 'p': p, 'at': 0})
    np.testing.assert_almost_equal(output["Y"], c)

    # test SequenceConstruct, SequenceErase, and SequenceLength
    seq_construct_node = helper.make_node('SequenceConstruct', ['a', 'b', 'c'],
                                          ['S'])
    seq_erase_node = helper.make_node('SequenceErase', ['S', 'p'], ['S1'])
    seq_length_node = helper.make_node('SequenceLength', ['S1'], ['Y'])

    graph = helper.make_graph(
        [seq_construct_node, seq_erase_node, seq_length_node],
        name='seq_construct_erase_length_test',
        inputs=[a_value_info, b_value_info, c_value_info, p_value_info],
        outputs=[out_value_info])
    model = helper.make_model(graph, producer_name='backend-test')
    tf_rep = prepare(model)
    output = tf_rep.run({'a': a, 'b': b, 'c': c, 'p': p})
    np.testing.assert_almost_equal(output["Y"], 2)

    # test SequenceConstruct and SequenceErase
    seq_construct_node = helper.make_node('SequenceConstruct', ['a', 'b', 'c'],
                                          ['S'])
    seq_erase_node = helper.make_node('SequenceErase', ['S', 'p'], ['S1'])
    seq_at_node = helper.make_node('SequenceAt', ['S1', 'at'], ['Y'])

    graph = helper.make_graph([seq_construct_node, seq_erase_node, seq_at_node],
                              name='seq_construct_erase_test',
                              inputs=[
                                  a_value_info, b_value_info, c_value_info,
                                  p_value_info, at_value_info
                              ],
                              outputs=[out_value_info])
    model = helper.make_model(graph, producer_name='backend-test')
    tf_rep = prepare(model)
    output = tf_rep.run({'a': a, 'b': b, 'c': c, 'p': p, 'at': 0})
    np.testing.assert_almost_equal(output["Y"], b)
    output = tf_rep.run({'a': a, 'b': b, 'c': c, 'p': p, 'at': 1})
    np.testing.assert_almost_equal(output["Y"], c)

    # test SequenceConstruct and ConcatFromSequence
    seq_construct_node = helper.make_node('SequenceConstruct', ['a', 'b', 'c'],
                                          ['S'])
    concat_from_seq_node = helper.make_node('ConcatFromSequence', ['S'], ['Y'],
                                            axis=1)
    a = np.array([[1, 2], [3, 4]]).astype(np.float32)
    b = np.array([[5, 6], [7, 8]]).astype(np.float32)
    c = np.array([[9, 10], [11, 12]]).astype(np.float32)
    a_value_info = helper.make_tensor_value_info('a', onnx.TensorProto.FLOAT,
                                                 [2, 2])
    b_value_info = helper.make_tensor_value_info('b', onnx.TensorProto.FLOAT,
                                                 [2, 2])
    c_value_info = helper.make_tensor_value_info('c', onnx.TensorProto.FLOAT,
                                                 [2, 2])

    graph = helper.make_graph([seq_construct_node, concat_from_seq_node],
                              name='seq_construct_concat_test',
                              inputs=[a_value_info, b_value_info, c_value_info],
                              outputs=[out_value_info])
    model = helper.make_model(graph, producer_name='backend-test')
    tf_rep = prepare(model)
    output = tf_rep.run({'a': a, 'b': b, 'c': c})
    d = np.concatenate((a, b, c), axis=1).astype(np.float32)
    np.testing.assert_almost_equal(output["Y"], d)

    # test SplitToSequence and SequenceAt
    a = np.array([[1, 2, 3, 4, 5, 6, 7], [11, 12, 13, 14, 15, 16, 17],
                  [21, 22, 23, 24, 25, 26, 27]]).astype(np.float32)
    b = np.int32([2, 1])
    seq_split_node = helper.make_node('SplitToSequence', ['a', 'b'], ['S'])
    seq_at_node = helper.make_node('SequenceAt', ['S', 'at'], ['Y'])
    a_value_info = helper.make_tensor_value_info('a', onnx.TensorProto.FLOAT,
                                                 [3, 7])
    b_value_info = helper.make_tensor_value_info('b', onnx.TensorProto.INT32,
                                                 [2])
    at_value_info = helper.make_tensor_value_info('at', onnx.TensorProto.INT32,
                                                  [])

    graph = helper.make_graph(
        [seq_split_node, seq_at_node],
        name='split_to_seq_test',
        inputs=[a_value_info, b_value_info, at_value_info],
        outputs=[out_value_info])
    model = helper.make_model(graph, producer_name='backend-test')
    tf_rep = prepare(model)
    output = tf_rep.run({'a': a, 'b': b, 'at': 1})
    np.testing.assert_almost_equal(output["Y"], np.split(a, [2, 3])[1])

    axis = 1
    seq_split_node = helper.make_node('SplitToSequence', ['a'], ['S'],
                                      axis=axis)
    seq_at_node = helper.make_node('SequenceAt', ['S', 'at'], ['Y'])
    at_value_info = helper.make_tensor_value_info('at', onnx.TensorProto.INT32,
                                                  [])

    graph = helper.make_graph([seq_split_node, seq_at_node],
                              name='split_to_seq_test',
                              inputs=[a_value_info, at_value_info],
                              outputs=[out_value_info])
    model = helper.make_model(graph, producer_name='backend-test')
    tf_rep = prepare(model)
    output = tf_rep.run({'a': a, 'at': 0})
    np.testing.assert_almost_equal(output["Y"], np.split(a, 7, axis=1)[0])

    seq_split_node = helper.make_node('SplitToSequence', ['a'], ['S'],
                                      keepdims=0)
    seq_at_node = helper.make_node('SequenceAt', ['S', 'at'], ['Y'])
    at_value_info = helper.make_tensor_value_info('at', onnx.TensorProto.INT32,
                                                  [])

    graph = helper.make_graph([seq_split_node, seq_at_node],
                              name='split_to_seq_test',
                              inputs=[a_value_info, at_value_info],
                              outputs=[out_value_info])
    model = helper.make_model(graph, producer_name='backend-test')
    tf_rep = prepare(model)
    output = tf_rep.run({'a': a, 'at': 0})
    expected = [np.squeeze(res) for res in np.split(a, 3)]
    np.testing.assert_almost_equal(output["Y"], expected[0])


if __name__ == '__main__':
  unittest.main()
