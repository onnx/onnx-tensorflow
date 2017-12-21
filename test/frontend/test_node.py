from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import tensorflow as tf

from onnx_tf.frontend import convert_graph
from onnx import helper

# for testing
from onnx_tf.backend import prepare

def get_node_by_name(nodes, name):
  for node in nodes:
    if node.name == name:
      return node

def get_rnd(shape, low=-1.0, high=1.0, dtype=np.float32):
  if (dtype==np.float32):
    return (np.random.uniform(low, high, np.prod(shape))
                     .reshape(shape)
                     .astype(np.float32))
  elif (dtype==np.int32):
    return (np.random.uniform(low, high, np.prod(shape))
                     .reshape(shape)
                     .astype(np.int32))
  elif dtype==np.bool_:
    return np.random.choice(a=[False, True], size=shape)

class TestNode(unittest.TestCase):
  """ Tests for nodes.
  Tests are dynamically added.
  Therefore edit test_cases to add more tests.
  """
  pass

def create_test(test_data):
  test_option = test_data[7] if len(test_data) > 7 else {}

  def do_test_expected(self):
    tf_op = test_data[1]
    output_name = test_data[2]
    inputs = test_data[3]
    attrs = test_data[4]
    device = test_data[5]
    channel_last = test_data[6]

    # Now construct input feed dict
    # keyed by input name
    onnx_feed_dict = {}
    # keyed by placeholder op
    tf_feed_dict = {}
    tf_param_list = []
    for idx, input_tensor in enumerate(inputs):
      if type(input_tensor) is np.ndarray:
        placeholder = tf.placeholder(input_tensor.dtype,
                                     shape=input_tensor.shape,
                                     name="in_" + str(idx))
        onnx_feed_dict["in_" + str(idx)] = input_tensor
        # TF have to get input in format : NHWC
        tf_feed_dict[placeholder] = input_tensor
        tf_param_list.append(placeholder)
      else:
        tf_param_list.append(input_tensor)
    test_op = tf_op(*tf_param_list, **attrs)
    tf_graph = tf.get_default_graph().as_graph_def(add_shapes=True)
    # Construct onnx graph, run with backend.
    output_node = get_node_by_name(tf_graph.node, output_name)
    onnx_graph = convert_graph(tf_graph, output_node, device=device,
            channel_last=channel_last)
    onnx_model = helper.make_model(onnx_graph)
    backend_rep = prepare(onnx_model, device=device, channel_last=channel_last)
    backend_output = backend_rep.run(onnx_feed_dict)[output_name]

    with tf.Session() as sess:
      tf_output = sess.run(test_op, tf_feed_dict)
    tf.reset_default_graph()

    # skip comparison if test_option specifies that
    # the test is call only.
    if (test_option.get("call_only", False)):
      return

    np.testing.assert_allclose(backend_output, tf_output)

  return do_test_expected

# organized as a tuple of the format:
# (test_name, tensorflow_op, output_node_name, LIST of inputs, MAP of attributes)
# Note that a python array is used differently than a numpy array
# in the sense that numpy array are passed in via tf.placeholder
# whereas python arrays are passed in as constants.
test_cases = [
("test_relu", tf.nn.relu, "Relu", [get_rnd([10, 10])], {}, 'CPU', True),
("test_or", tf.logical_or, "LogicalOr", [get_rnd([10, 10], dtype=np.bool_), get_rnd([10, 10], dtype=np.bool_)], {}, 'CPU', True),
("test_pow", tf.pow, "Pow", [get_rnd([10, 10]), get_rnd([10, 10])], {}, 'CPU', True),
("test_pad", tf.pad, "Pad", [get_rnd([2, 3]), [[1, 1,], [2, 2]]], {"mode": "constant"}, 'CPU', True),
("test_random_normal", tf.random_normal, "random_normal/RandomStandardNormal", [], {"shape": [1, 100, 100, 1], "mean": 0.0, "stddev": 1.0, "dtype": tf.float32, "seed": 42}, 'CPU', True, {"call_only": True}),
("test_random_normal", tf.random_normal, "random_normal/RandomStandardNormal", [], {"shape": [100, 100], "mean": 0.0, "stddev": 1.0, "dtype": tf.float32, "seed": 42}, 'CPU', True, {"call_only": True}),
("test_random_uniform", tf.random_uniform, "random_uniform", [], {"shape": [100, 100], "minval": 0.0, "maxval": 1.0, "dtype": tf.float32, "seed": 42}, 'CPU', True, {"call_only": True}),
("test_random_uniform", tf.random_uniform, "random_uniform", [], {"shape": [1, 42, 42, 3], "minval": 0.0, "maxval": 1.0, "dtype": tf.float32, "seed": 42}, 'CPU', True, {"call_only": True}),
("test_reciprocal", tf.reciprocal, "Reciprocal", [get_rnd([10, 10])], {}, 'CPU', True),
("test_reduce_max", tf.reduce_max, "Max", [get_rnd([10, 10, 10, 10])], {"axis": [0, 1, 2, 3], "keep_dims": True}, 'CPU', True),
("test_reduce_max", tf.reduce_max, "Max", [get_rnd([10, 10])], {"keep_dims": True}, 'CPU', True),
("test_reduce_mean", tf.reduce_mean, "Mean", [get_rnd([10, 10])], {"keep_dims": True}, 'CPU', True),
("test_reduce_min", tf.reduce_min, "Min", [get_rnd([10, 10])], {"keep_dims": True}, 'CPU', True),
("test_reduce_prod", tf.reduce_prod, "Prod", [get_rnd([10, 10])], {"keep_dims": True}, 'CPU', True),
("test_reduce_prod", tf.reduce_prod, "Prod", [get_rnd([1, 10, 10, 3])], {"axis": [1, 2], "keep_dims": True}, 'CPU', True),
("test_reduce_sum", tf.reduce_sum, "Sum", [get_rnd([10, 10])], {"keep_dims": True}, 'CPU', True),
("test_reduce_sum", tf.reduce_sum, "Sum", [get_rnd([1, 10, 10, 3])], {"axis": [1, 2], "keep_dims": True}, 'CPU', True),
("test_reshape", tf.reshape, "Reshape", [get_rnd([1, 16, 16, 3]), [1, 16*16, 3]], {}, 'CPU', True),
("test_reshape", tf.reshape, "Reshape", [get_rnd([1, 16, 16, 3]), [3, 16*16]], {}, 'CPU', True),
("test_reshape", tf.reshape, "Reshape", [get_rnd([1, 16, 16, 3]), [3, 16, 16]], {}, 'CPU', True),
("test_reshape", tf.reshape, "Reshape", [get_rnd([1, 16, 16, 1]), [16, 16]], {}, 'CPU', True),
("test_reshape", tf.reshape, "Reshape", [get_rnd([10, 10]), [4, 25]], {}, 'CPU', True),
("test_sigmoid", tf.sigmoid, "Sigmoid", [get_rnd([10, 10])], {}, 'CPU', True),
("test_split", tf.split, "split", [get_rnd([1, 10, 10, 3]), [1, 5, 5, 1]], {}, 'CPU', True),
("test_split", tf.split, "split", [get_rnd([10, 10]), [5, 5]], {}, 'CPU', True),
("test_sqrt", tf.sqrt, "Sqrt", [get_rnd([10, 10])], {}, 'CPU', True),
("test_squeeze", tf.squeeze, "Squeeze", [get_rnd([1, 1, 1, 1])], {"axis":[0, 3, 2, 1]}, 'CPU', True),
("test_squeeze", tf.squeeze, "Squeeze", [get_rnd([1, 10, 10])], {"axis":[0]}, 'CPU', True),
("test_squeeze", tf.squeeze, "Squeeze", [get_rnd([1, 10, 10, 1])], {"axis":[0, 3]}, 'CPU', True),
("test_squeeze", tf.squeeze, "Squeeze", [get_rnd([10, 1, 1, 3])], {"axis":[1, 2]}, 'CPU', True),
("test_subtract", tf.subtract, "Sub", [get_rnd([10, 10]), get_rnd([10, 10])], {}, 'CPU', True),
("test_tanh", tf.tanh, "Tanh", [get_rnd([10, 10])], {}, 'CPU', True),
("test_xor", tf.logical_xor, "LogicalXor", [get_rnd([10, 10], dtype=np.bool_), get_rnd([10, 10], dtype=np.bool_)], {}, 'CPU', True),
("test_transpose", tf.transpose, "transpose", [get_rnd([1, 3, 2, 10])], {"perm":[2, 3, 1, 0]}, 'CPU', True),
("test_transpose", tf.transpose, "transpose", [get_rnd([1, 16, 16, 3])], {"perm":[0, 3, 1, 2]}, 'CPU', True),
("test_transpose", tf.transpose, "transpose", [get_rnd([1, 2, 2, 10])], {"perm":[3, 1, 2, 0]}, 'CPU', True),
("test_transpose", tf.transpose, "transpose", [get_rnd([2, 10])], {"perm":[1, 0]}, 'CPU', True),
("test_concat", tf.concat, "concat", [[get_rnd([2, 1]), get_rnd([2, 2]), get_rnd([2, 1])], 1], {}, 'CPU', True),
("test_concat", tf.concat, "concat", [[get_rnd([2, 1, 2]), get_rnd([2, 2, 2]), get_rnd([2, 1, 2])], 1], {}, 'CPU', True),
("test_concat", tf.concat, "concat", [[get_rnd([1, 10, 10, 1]), get_rnd([1, 10, 10, 1]), get_rnd([1, 10, 10, 1])], 3], {}, 'CPU', True),
("test_concat", tf.concat, "concat", [[get_rnd([1, 10, 10, 1]), get_rnd([1, 10, 10, 1]), get_rnd([1, 10, 10, 1])], 0], {}, 'CPU', True),
]

for k, val in enumerate(test_cases):
    test_method = create_test(val)
    test_method.__name__ = str(val[0])
    setattr (TestNode, test_method.__name__, test_method)

if __name__ == '__main__':
  unittest.main()
