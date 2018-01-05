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

from tensorflow.python.client import device_lib
DEVICE_TO_TEST = ["CPU", "CUDA"]

def supports_device(device):
  if device == "CUDA":
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in
                local_device_protos if x.device_type == 'GPU']) > 0
  elif device == "CPU":
    return True
  return False

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
  test_option = test_data[5] if len(test_data) > 5 else {}

  def do_test_expected(self):
    tf_op = test_data[1]
    output_name = test_data[2]
    inputs = test_data[3]
    attrs = test_data[4]
    for device in DEVICE_TO_TEST:
      if not supports_device(device):
        # This case can not be tester on this device
        continue
      for channel_last in [True, False]:
        if channel_last in [False] and device in ["CPU"]:
          # This combination is incompatible
          continue

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
("test_relu0", tf.nn.relu, "Relu", [get_rnd([10, 10])], {}),
("test_or0", tf.logical_or, "LogicalOr", [get_rnd([10, 10], dtype=np.bool_), get_rnd([10, 10], dtype=np.bool_)], {}),
("test_pow0", tf.pow, "Pow", [get_rnd([10, 10]), get_rnd([10, 10])], {}),
("test_pad1", tf.pad, "Pad", [get_rnd([2, 3]), [[1, 1,], [2, 2]]], {"mode": "constant"}),
("test_pad0", tf.pad, "Pad", [get_rnd([10, 2, 2, 3]), [[0,0], [2, 2], [2, 2], [0,0]]], {"mode": "constant"}),
("test_random_normal1", tf.random_normal, "random_normal/RandomStandardNormal", [], {"shape": [1, 100, 100, 1], "mean": 0.0, "stddev": 1.0, "dtype": tf.float32, "seed": 42}, {"call_only": True}),
("test_random_normal0", tf.random_normal, "random_normal/RandomStandardNormal", [], {"shape": [100, 100], "mean": 0.0, "stddev": 1.0, "dtype": tf.float32, "seed": 42}, {"call_only": True}),
("test_random_uniform1", tf.random_uniform, "random_uniform", [], {"shape": [100, 100], "minval": 0.0, "maxval": 1.0, "dtype": tf.float32, "seed": 42}, {"call_only": True}),
("test_random_uniform0", tf.random_uniform, "random_uniform", [], {"shape": [1, 42, 42, 3], "minval": 0.0, "maxval": 1.0, "dtype": tf.float32, "seed": 42}, {"call_only": True}),
("test_reciprocal0", tf.reciprocal, "Reciprocal", [get_rnd([10, 10])], {}),
("test_reduce_max1", tf.reduce_max, "Max", [get_rnd([10, 10, 10, 10])], {"axis": [0, 1, 2, 3], "keep_dims": True}),
("test_reduce_max0", tf.reduce_max, "Max", [get_rnd([10, 10])], {"keep_dims": True}),
("test_reduce_mean0", tf.reduce_mean, "Mean", [get_rnd([10, 10])], {"keep_dims": True}),
("test_reduce_min0", tf.reduce_min, "Min", [get_rnd([10, 10])], {"keep_dims": True}),
("test_reduce_prod1", tf.reduce_prod, "Prod", [get_rnd([10, 10])], {"keep_dims": True}),
("test_reduce_prod0", tf.reduce_prod, "Prod", [get_rnd([1, 10, 10, 3])], {"axis": [1, 2], "keep_dims": True}),
("test_reduce_sum1", tf.reduce_sum, "Sum", [get_rnd([10, 10])], {"keep_dims": True}),
("test_reduce_sum0", tf.reduce_sum, "Sum", [get_rnd([1, 10, 10, 3])], {"axis": [1, 2], "keep_dims": True}),
("test_reshape2", tf.reshape, "Reshape", [get_rnd([1, 16, 16, 3]), [1, 16*16, 3]], {}),
("test_reshape1", tf.reshape, "Reshape", [get_rnd([1, 16, 16, 1]), [1, 16, 16, 1]], {}),
("test_reshape0", tf.reshape, "Reshape", [get_rnd([10, 10]), [4, 25]], {}),
("test_sigmoid0", tf.sigmoid, "Sigmoid", [get_rnd([10, 10])], {}),
("test_split4", tf.split, "split", [get_rnd([1, 10, 10, 3]), [1, 1, 1]], {"axis":3}),
("test_split2", tf.split, "split", [get_rnd([1, 10, 10, 3]), [5, 5]], {"axis":2}),
("test_split1", tf.split, "split", [get_rnd([10, 10]), [5, 5]], {"axis":1}),
("test_split0", tf.split, "split", [get_rnd([10, 10]), [5, 5]], {"axis":0}),
("test_sqrt0", tf.sqrt, "Sqrt", [get_rnd([10, 10])], {}),
("test_squeeze6", tf.squeeze, "Squeeze", [get_rnd([1, 1, 1, 10])], {}),
("test_squeeze5", tf.squeeze, "Squeeze", [get_rnd([10, 1, 1, 1])], {}),
("test_squeeze4", tf.squeeze, "Squeeze", [get_rnd([1, 1, 10, 1])], {}),
("test_squeeze3", tf.squeeze, "Squeeze", [get_rnd([10, 1, 1, 1])], {"axis":[1, 2]}),
("test_squeeze2", tf.squeeze, "Squeeze", [get_rnd([1, 10, 10])], {"axis":[0]}),
("test_squeeze1", tf.squeeze, "Squeeze", [get_rnd([1, 10, 10, 1])], {"axis":[0, 3]}),
("test_squeeze0", tf.squeeze, "Squeeze", [get_rnd([10, 1, 1, 3])], {"axis":[1, 2]}),
("test_subtract0", tf.subtract, "Sub", [get_rnd([10, 10]), get_rnd([10, 10])], {}),
("test_tanh0", tf.tanh, "Tanh", [get_rnd([10, 10])], {}),
("test_xor0", tf.logical_xor, "LogicalXor", [get_rnd([10, 10], dtype=np.bool_), get_rnd([10, 10], dtype=np.bool_)], {}),
("test_transpose3", tf.transpose, "transpose", [get_rnd([1, 3, 2, 10])], {"perm":[2, 3, 1, 0]}),
("test_transpose2", tf.transpose, "transpose", [get_rnd([1, 16, 16, 3])], {"perm":[0, 3, 1, 2]}),
("test_transpose1", tf.transpose, "transpose", [get_rnd([1, 2, 2, 10])], {"perm":[3, 1, 2, 0]}),
("test_transpose0", tf.transpose, "transpose", [get_rnd([2, 10])], {"perm":[1, 0]}),
("test_concat3", tf.concat, "concat", [[get_rnd([2, 1]), get_rnd([2, 2]), get_rnd([2, 1])], 1], {}),
("test_concat2", tf.concat, "concat", [[get_rnd([2, 1, 2]), get_rnd([2, 2, 2]), get_rnd([2, 1, 2])], 1], {}),
("test_concat1", tf.concat, "concat", [[get_rnd([1, 10, 10, 1]), get_rnd([1, 10, 10, 1]), get_rnd([1, 10, 10, 1])], 3], {}),
("test_concat0", tf.concat, "concat", [[get_rnd([1, 10, 10, 1]), get_rnd([1, 10, 10, 1]), get_rnd([1, 10, 10, 1])], 0], {}),
]

for k, val in enumerate(test_cases):
    test_method = create_test(val)
    test_method.__name__ = str(val[0])
    setattr (TestNode, test_method.__name__, test_method)

if __name__ == '__main__':
  unittest.main()
