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

class TestNode(unittest.TestCase):
  """ Tests for nodes.
  Tests are dynamically added.
  Therefore edit test_cases to add more tests.
  """
  pass

def get_node_by_name(nodes, name):
    for node in nodes:
        if node.name == name:
          return node

def get_rnd(shape, low=-1.0, high=1.0):
    return (np.random.uniform(low, high, np.prod(shape))
                     .reshape(shape)
                     .astype(np.float32))

def create_test(test_data):
  def do_test_expected(self):
    tf_op = test_data[1]
    output_name = test_data[2]
    inputs = test_data[3]

    # Now construct input feed dict
    # keyed by input name
    onnx_feed_dict = {}
    # keyed by placeholder op
    tf_feed_dict = {}
    tf_param_list = []
    for idx, input_tensor in enumerate(inputs):
      placeholder = tf.placeholder(tf.float32,
                                   shape=input_tensor.shape,
                                   name="in_" + str(idx))
      onnx_feed_dict["in_" + str(idx)] = input_tensor
      tf_feed_dict[placeholder] = input_tensor
      tf_param_list.append(placeholder)

    test_op = tf_op(*tf_param_list)
    tf_graph = test_op.graph.as_graph_def(add_shapes=True)

    # Construct onnx graph, run with backend.
    output_node = get_node_by_name(tf_graph.node, output_name)
    onnx_graph = convert_graph(tf_graph, output_node)
    backend_rep = prepare(helper.make_model(onnx_graph))
    backend_output = backend_rep.run(onnx_feed_dict)[output_name]

    with tf.Session() as sess:
      tf_output = sess.run(test_op, tf_feed_dict)

    np.testing.assert_allclose(backend_output, tf_output)
  return do_test_expected

# organized as a tuple of the format:
# (test_name, tensorflow_op, output_node_name, list of inputs)

test_cases = [
("test_relu", tf.nn.relu, "Relu", [get_rnd([10, 10])])
]

for k, val in enumerate(test_cases):
    test_method = create_test(val)
    test_method.__name__ = str(val[0])
    setattr (TestNode, test_method.__name__, test_method)

if __name__ == '__main__':
  unittest.main()