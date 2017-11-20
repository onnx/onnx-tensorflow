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
  """ Tests for nodes
  """
  def _get_rnd(self, shape, low=-1.0, high=1.0):
    return np.random.uniform(low, high, np.prod(shape)) \
                    .reshape(shape) \
                    .astype(np.float32)

  def test_relu(self):
    shape = (10, 10)
    x = tf.placeholder(tf.float32, shape=shape)
    y = tf.nn.relu(x)

    tf_graph = y.graph.as_graph_def(add_shapes=True)
    for node in tf_graph.node:
        if node.name == "Relu":
            output_node = node

    onnx_graph = convert_graph(tf_graph, output_node)
    tf_rep = prepare(helper.make_model(onnx_graph))
    in_tensor = self._get_rnd(shape)
    output = tf_rep.run({"Placeholder": in_tensor})["Relu"]
    with tf.Session() as sess:
        output_tf = sess.run(y, feed_dict={
            x: in_tensor
            })

    np.testing.assert_allclose(output, output_tf)

if __name__ == '__main__':
  unittest.main()