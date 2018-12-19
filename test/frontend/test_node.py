from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import tensorflow as tf
from collections import Iterable

from onnx_tf.frontend import tensorflow_graph_to_onnx_model
from onnx import checker

# for testing
from onnx_tf.backend import prepare
from onnx_tf.common.legacy import legacy_opset_pre_ver


def get_node_by_name(nodes, name):
  for node in nodes:
    if node.name == name:
      return node


def get_rnd(shape, low=-1.0, high=1.0, dtype=np.float32):
  if dtype == np.float32:
    return (np.random.uniform(low, high,
                              np.prod(shape)).reshape(shape).astype(np.float32))
  elif dtype == np.int32:
    return (np.random.uniform(low, high,
                              np.prod(shape)).reshape(shape).astype(np.int32))
  elif dtype == np.bool_:
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
    tf.reset_default_graph()
    tf_op = test_data[1]
    output_name = test_data[2]
    inputs = test_data[3]
    attrs = test_data[4]

    # Now construct input feed dict
    # keyed by input name
    onnx_feed_dict = {}
    # keyed by placeholder op
    tf_feed_dict = {}
    tf_param_list = []
    for idx, input_tensor in enumerate(inputs):
      if type(input_tensor) is np.ndarray:
        placeholder = tf.placeholder(
            input_tensor.dtype, shape=input_tensor.shape, name="in_" + str(idx))
        onnx_feed_dict["in_" + str(idx)] = input_tensor
        tf_feed_dict[placeholder] = input_tensor
        tf_param_list.append(placeholder)
      else:
        tf_param_list.append(input_tensor)
    test_op = tf_op(*tf_param_list, **attrs)
    tf_graph = tf.get_default_graph().as_graph_def(add_shapes=True)
    # Construct onnx graph, run with backend.
    onnx_model = tensorflow_graph_to_onnx_model(
        tf_graph,
        output_name,
        ignore_unimplemented=test_option.get("ignore_unimplemented", False))
    if not test_option.get("ignore_unimplemented", False):
      checker.check_model(onnx_model)
      backend_rep = prepare(onnx_model)
      backend_output = []
      backend_rep_outputs = backend_rep.run(onnx_feed_dict)
      for output in backend_rep.outputs:
        backend_output.append(backend_rep_outputs[output])
      backend_output = np.asarray(backend_output)
      backend_output = np.squeeze(
          backend_output, 0) if backend_output.shape[0] == 1 else backend_output

      with tf.Session() as sess:
        tf_output = sess.run(test_op, tf_feed_dict)

      # make sure backend_output and tf_output are Iterable
      if backend_output.ndim == 0:
        backend_output = backend_output.reshape(1)
      if isinstance(tf_output, Iterable) == False:
        tf_output = [tf_output]

      # skip comparison if test_option specifies that
      # the test is call only.
      if test_option.get("call_only", False):
        return
      for backend_o, tf_o in zip(backend_output, tf_output):
        np.testing.assert_allclose(backend_o, tf_o, rtol=1e-3, atol=1e-7)

  return do_test_expected


# yapf: disable
# organized as a tuple of the format:
# (test_name, tensorflow_op, output_node_name, LIST of inputs, MAP of attributes)
# Note that a python array is used differently than a numpy array
# in the sense that numpy array are passed in via tf.placeholder
# whereas python arrays are passed in as constants.
test_cases = [
("test_abs", tf.abs, "Abs", [get_rnd([1, 2, 3, 4])], {}),
("test_arg_max", tf.argmax, "ArgMax", [get_rnd([1, 2, 3, 4])], {"axis": -1}),
("test_arg_min", tf.argmin, "ArgMin", [get_rnd([1, 2, 3, 4])], {"axis": -1}),
("test_cast", tf.cast, "Cast", [get_rnd([10, 10]), tf.float16], {}),
("test_size", tf.size, "Size", [get_rnd([5, 5])], {}),
("test_ceil", tf.ceil, "Ceil", [get_rnd([10, 10], -10, 10)], {}),
("test_constant_fill", tf.fill, "Fill", [[1, 2, 3], 1], {}),
("test_exp", tf.exp, "Exp", [get_rnd([10, 10])], {}),
("test_expand_dims", tf.expand_dims, "ExpandDims", [get_rnd([1, 2, 3, 4])], {"axis": 1}),
("test_floor", tf.floor, "Floor", [get_rnd([10, 10], -10, 10)], {}),
("test_floor_div", tf.floordiv, "floordiv", [get_rnd([10, 10], -10, 10), get_rnd([10, 10], -10, 10)], {}),
("test_gatherV2", tf.gather, "GatherV2", [get_rnd([3, 3]), [0, 2]], {"axis": 1}),
("test_identity", tf.identity, "Identity", [get_rnd([10, 10])], {}),
("test_log", tf.log, "Log", [get_rnd([10, 10])], {}),
("test_log_softmax", tf.nn.log_softmax, "LogSoftmax", [get_rnd([10, 10])], {}),
("test_or", tf.logical_or, "LogicalOr", [get_rnd([10, 10], dtype=np.bool_), get_rnd([10, 10], dtype=np.bool_)], {}),
("test_pow", tf.pow, "Pow", [get_rnd([10, 10]), get_rnd([10, 10])], {}),
("test_pack", tf.stack, "stack_1", [[get_rnd([3, 4]), get_rnd([3, 4])]], {}),
("test_pad", tf.pad, "Pad", [get_rnd([2, 3]), [[1, 1,], [2, 2]]], {"mode": "constant"}),
("test_random_normal", tf.random_normal, "random_normal/RandomStandardNormal", [], {"shape": [100, 100], "mean": 0.0, "stddev": 1.0, "dtype": tf.float32, "seed": 42}, {"call_only": True}),
("test_random_uniform", tf.random_uniform, "random_uniform", [], {"shape": [100, 100], "minval": 0.0, "maxval": 1.0, "dtype": tf.float32, "seed": 42}, {"call_only": True}),
("test_reciprocal", tf.reciprocal, "Reciprocal", [get_rnd([10, 10])], {}),
("test_reduce_max", tf.reduce_max, "Max", [get_rnd([10, 10])], {"keep_dims": True}),
("test_reduce_mean", tf.reduce_mean, "Mean", [get_rnd([10, 10])], {"keep_dims": True}),
("test_reduce_min", tf.reduce_min, "Min", [get_rnd([10, 10])], {"keep_dims": True}),
("test_reduce_prod", tf.reduce_prod, "Prod", [get_rnd([10, 10])], {"keep_dims": True}),
("test_reduce_sum", tf.reduce_sum, "Sum", [get_rnd([10, 10])], {"keep_dims": True}),
("test_reduce_sum_scalar_axes", tf.reduce_sum, "Sum", [get_rnd([10, 10]), 0], {"keep_dims": True}),
("test_relu", tf.nn.relu, "Relu", [get_rnd([10, 10])], {}),
("test_relu6", tf.nn.relu6, "Relu6", [get_rnd([10, 10])], {}),
("test_reshape", tf.reshape, "Reshape", [get_rnd([10, 10]), [4, 25]], {}),
("test_rsqrt", tf.rsqrt, "Rsqrt", [get_rnd([3, 3])], {}),
("test_selu", tf.nn.selu, "Selu", [get_rnd([10, 10])], {}),
("test_shape", tf.shape, "Shape", [get_rnd([1, 2, 3, 4])], {}),
("test_sigmoid", tf.sigmoid, "Sigmoid", [get_rnd([10, 10])], {}),
("test_slice", tf.slice, "Slice", [get_rnd([5, 6, 7])], {"begin": [1, 0, 0], "size": [1, 1, 3]}),
("test_softmax", tf.nn.softmax, "Softmax", [get_rnd([10, 10])], {}),
("test_softplus", tf.nn.softplus, "Softplus", [get_rnd([10, 10])], {}),
("test_softsign", tf.nn.softsign, "Softsign", [get_rnd([10, 10])], {}),
("test_space_to_depth", tf.space_to_depth, "SpaceToDepth", [get_rnd([2, 8, 8, 5])], {"block_size": 2}),
("test_split_v", tf.split, "split", [get_rnd([10, 10]), [2, 3, 5]], {}),
("test_split", tf.split, "split", [get_rnd([10, 10]), 2], {}),
("test_sqrt", tf.sqrt, "Sqrt", [get_rnd([10, 10])], {}),
("test_square", tf.square, "Square", [get_rnd([10, 10])], {}),
("test_squeeze", tf.squeeze, "Squeeze", [get_rnd([1, 1, 10, 10])], {"axis":[0, 1]}),
("test_subtract", tf.subtract, "Sub", [get_rnd([10, 10]), get_rnd([10, 10])], {}),
("test_tanh", tf.tanh, "Tanh", [get_rnd([10, 10])], {}),
("test_top_k", tf.nn.top_k, "TopKV2", [get_rnd([10, 10, 10, 10])], {"k": 3}),
# Use reverse to test ignore_unimplemented
("test_unimplemented", tf.reverse, "ReverseV2", [get_rnd([1, 2, 3, 4]), [3]], {}, {"ignore_unimplemented": True}),
("test_unpack", tf.unstack, "unstack", [get_rnd([2, 3, 4])], {}),
("test_xor", tf.logical_xor, "LogicalXor", [get_rnd([10, 10], dtype=np.bool_), get_rnd([10, 10], dtype=np.bool_)], {}),
("test_transpose", tf.transpose, "transpose", [get_rnd([2, 10])], {"perm":[1, 0]}),
("test_concat", tf.concat, "concat", [[get_rnd([1, 10]),get_rnd([10, 10]),get_rnd([20, 10])], 0], {}),
("test_bias_add_nchw", tf.nn.bias_add, "BiasAdd", [get_rnd([10, 32, 10, 10]),get_rnd([32])], {"data_format":"NCHW"}),
("test_bias_add_nhwc", tf.nn.bias_add, "BiasAdd", [get_rnd([10, 10, 10, 32]),get_rnd([32])], {"data_format":"NHWC"}),
]

if not legacy_opset_pre_ver(6):
  test_cases.append(("test_tile", tf.tile, "Tile", [get_rnd([1, 2, 3, 4]), np.random.randint(1, 10, (4,), dtype=np.int32)], {}))

if not legacy_opset_pre_ver(9):
  test_cases.append(("test_strided_slice", tf.strided_slice, "StridedSlice", [get_rnd([5, 5]), [0, 0], [1, 5], [1, 1]], {}))
  test_cases.append(("test_strided_slice_shrink", tf.strided_slice, "StridedSlice", [get_rnd([5, 5]), [0, 0], [1, 3], [1, 1]], {"shrink_axis_mask":1}))
  test_cases.append(("test_resize_bilinear", tf.image.resize_bilinear, "ResizeBilinear", [get_rnd([2, 5, 5, 8]), [10, 10]], {}))
  test_cases.append(("test_sinh", tf.sinh, "Sinh", [get_rnd([10, 10])], {}))
  test_cases.append(("test_cosh", tf.cosh, "Cosh", [get_rnd([10, 10])], {}))
  test_cases.append(("test_asinh", tf.asinh, "Asinh", [get_rnd([10, 10])], {}))
  test_cases.append(("test_acosh", tf.acosh, "Acosh", [get_rnd([10, 10])], {}))
  test_cases.append(("test_tanh", tf.tanh, "Tanh", [get_rnd([10, 10])], {}))

# yapf: enable

for k, val in enumerate(test_cases):
  test_method = create_test(val)
  test_method.__name__ = str(val[0])
  setattr(TestNode, test_method.__name__, test_method)

if __name__ == '__main__':
  unittest.main()
