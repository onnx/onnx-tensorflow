"""Backend for running ONNX on Tensorflow

To run this, you will need to have Tensorflow installed as well.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import re
import warnings

import numpy as np
from onnx import checker
from onnx.onnx_pb2 import GraphProto, TensorProto, AttributeProto
from onnx_tf.tf_net import TensorflowNet
from onnx_tf.backend_rep import TensorflowRep
import onnx.numpy_helper
import onnx.defs

from onnx.backend.base import (
    Backend,
    BackendRep,
    Device,
    DeviceType,
    namedtupledict,
)

from onnx import onnx_pb2, helper
import tensorflow as tf
from tensorflow.python.client import device_lib

# TODO: allow more flexible placement
def get_device_option(device):
  m = {DeviceType.CPU: '/cpu',
       DeviceType.CUDA: '/gpu'}
  return m[device.type]

# TODO: Move this into ONNX main library
def convertAttributeProto(onnx_arg):
  """
  Convert an ONNX AttributeProto into an appropriate Python object
  for the type.
  NB: Tensor attribute gets returned as the straight proto.
  """
  if onnx_arg.HasField('f'):
    return onnx_arg.f
  elif onnx_arg.HasField('i'):
    return onnx_arg.i
  elif onnx_arg.HasField('s'):
    return onnx_arg.s
  elif onnx_arg.HasField('t'):
    return onnx_arg.t  # this is a proto!
  elif onnx_arg.floats:
    return list(onnx_arg.floats)
  elif onnx_arg.ints:
    return list(onnx_arg.ints)
  elif onnx_arg.strings:
    return list(onnx_arg.strings)
  else:
    raise ValueError("Unsupported ONNX attribute: {}".format(onnx_arg))

class OnnxAttributes(dict):
  """
  This is a more convenient way to work with ONNX/Caffe2 attributes
  that is not the protobuf representation.
  """
  @staticmethod
  def from_onnx(args):
    d = OnnxAttributes()
    for arg in args:
      d[arg.name] = convertAttributeProto(arg)
    return d

  def caffe2(self, kmap=lambda x: x):
    for k, v in self.items():
      yield caffe2.python.utils.MakeArgument(kmap(k), v)

# TODO: Move this into ONNX main library
class OnnxNode(object):
  """
  Reimplementation of NodeProto from ONNX, but in a form
  more convenient to work with from Python.
  We may temporarily edit these nodes to get them into Caffe2 form,
  before actually translating into the Caffe2 protobuf, since this
  is easier than decomposing everything, and putting it back together
  when we're ready.
  """
  def __init__(self, node):
    self.name = str(node.name)
    self.op_type = str(node.op_type)
    self.attrs = OnnxAttributes.from_onnx(node.attribute)
    self.consumed_inputs = self.attrs.pop("consumed_inputs", None)
    self.inputs = list(node.input)
    self.outputs = list(node.output)

class TensorflowBackend(Backend):
  """ Tensorflow Backend for ONNX
  """

  onnx_tf_attribute_map = {
      "scale": "stddev",
      "high": "maxval",
      "low": "minval",
      "axes": "axis",
      "keepdims": "keep_dims",
      "axis": "dim",
      "to": "dtype",
  }

  onnx_tf_per_op_attr_map = {}

  onnx_tf_op_map = {
      "abs": tf.abs,
      "cast": tf.cast,
      "ceil": tf.ceil,
      "relu": tf.nn.relu,
      "dot": tf.contrib.keras.backend.dot,
      "exp": tf.exp,
      "floor": tf.floor,
      "gather": tf.gather,
      "log": tf.log,
      "neg": tf.negative,
      "pow": tf.pow,
      "random_normal": tf.random_normal,
      "random_uniform": tf.random_uniform,
      "reciprocal": tf.reciprocal,
      "reduce_log_sum_exp": tf.reduce_logsumexp,
      "reduce_max": tf.reduce_max,
      "reduce_mean": tf.reduce_mean,
      "reduce_min": tf.reduce_min,
      "reduce_prod": tf.reduce_prod,
      "reduce_sum": tf.reduce_sum,
      "sigmoid": tf.sigmoid,
      # default parameter
      "softmax": tf.nn.softmax,
      "sqrt": tf.sqrt,
      "squeeze": tf.squeeze,
      "tanh": tf.tanh,
      "transpose": tf.transpose,
  }

  tensor_type_to_tf_type = {
      TensorProto.FLOAT: tf.float32,
      TensorProto.UINT8: tf.uint8,
      TensorProto.INT8: tf.int8,
      TensorProto.UINT16: tf.uint16,
      TensorProto.INT16: tf.int16,
      TensorProto.INT32: tf.int32,
      TensorProto.INT64: tf.int64,
      TensorProto.BOOL: tf.bool,
      TensorProto.FLOAT16: tf.float16,
      TensorProto.DOUBLE: tf.float64,
      TensorProto.COMPLEX64: tf.complex64,
      TensorProto.COMPLEX128: tf.complex128,
      # TODO: uncomment this in the future
      # TensorProto.UINT32: tf.uint32,
      # TensorProto.UINT64: tf.uint64,
  }

  tensor_type_enum = [
      "undefined",
      tf.float32,
      tf.uint8,
      tf.int8,
      tf.uint16,
      tf.int16,
      tf.int32,
      tf.int64,
      tf.bool,
      tf.float16,
      tf.float64,
      tf.complex64,
      tf.complex128,
      # TODO: uncomment this in the future
      # tf.uint32,
      # tf.uint64,
  ]

  type_string_to_tf_type = {
      "float": tf.float32,
      "uint8": tf.uint8,
      "int8": tf.int8,
      "uint16": tf.uint16,
      "int16": tf.int16,
      "int32": tf.int32,
      "int64": tf.int64,
      "bool": tf.bool,
      "float16": tf.float16,
      "double": tf.float64,
      "complex64": tf.complex64,
      "complex128": tf.complex128,
      # TODO: uncomment this in the future
      # "uint32": tf.uint32,
      # "uint64": tf.uint64,
  }

  attr_translator = {
      "dtype": lambda cls, x: cls.tensor_type_to_tf_type[x],
      "keepdims": lambda cls, x: bool(x),
      "to": lambda cls, x: cls.type_string_to_tf_type[x],
  }

  @classmethod
  def run_node(cls, node, inputs, device='CPU'):
    super(TensorflowBackend, cls).run_node(node, inputs, device)
    node = OnnxNode(node)
    device_option = get_device_option(Device(device))
    input_tensors = []
    for i in inputs:
      input_tensors.append(tf.constant(i))

    if isinstance(inputs, dict):
      feed_dict_raw = inputs
    else:
      assert len(node.inputs) == len(inputs)
      feed_dict_raw = dict(zip(node.inputs, inputs))
    # TODO: is constant the best way for feeding inputs?
    input_dict = dict([(x[0], tf.constant(x[1])) for x in \
                       feed_dict_raw.items()])
    ops = cls._onnx_node_to_tensorflow_op(node, input_dict)
    output_vals = []
    with tf.Session() as sess:
      with tf.device(device_option):
        sess.run(tf.initialize_all_variables())
        output_vals = sess.run(ops)

    return namedtupledict('Outputs', node.outputs)(*output_vals)

  @classmethod
  def onnx_graph_to_tensorflow_net(cls, graph_def):
    # initializer: TensorProtos representing the values to initialize
    # a given tensor.
    # initialized: A list of names of the initialized tensors.
    if graph_def.initializer:
      input_dict_items = cls.onnx_initializer_to_input_dict_items(
        graph_def.initializer)
      initialized = {init.name for init in graph_def.initializer}
    else:
      input_dict_items = []
      initialized = set()

    predict_net = TensorflowNet()
    predict_net.name = graph_def.name

    predict_net.external_input.extend(
      value_info.name for value_info in graph_def.input)
    predict_net.external_output.extend(
      value_info.name for value_info in graph_def.output)

    # creating placeholders for currently unkown inputs
    for value_info in graph_def.input:
      if value_info.name in initialized:
          continue

      shape = list(d.dim_value for d in \
        value_info.type.tensor_type.shape.dim)
      x = tf.placeholder(cls.tensor_type_enum[value_info.type.tensor_type.elem_type],
                         name=value_info.name, shape=shape)
      input_dict_items.append([value_info.name, x])

    # input dict: this dictionary is a map from variable names
    # to the latest produced tensors of the given name.
    # This dictionary will get updated as build the graph because
    # some ops may produce a result tensor with the same name as
    # the input tensor. The input dict tracks the latest produced
    # tensors.
    input_dict = dict(input_dict_items)
    # Since input dict may be updated, we need to keep a copy
    # of the original input dict where we track the earliest
    # defined tensors so we can have access to the placeholders
    # to feed in input tensors when we run the graph.
    original_input_dict = dict(input_dict_items)
    output_dict = dict()

    for node in graph_def.node:
      node = OnnxNode(node)
      output_ops = cls._onnx_node_to_tensorflow_op(node, input_dict)
      curr_node_output_map = zip(node.outputs, output_ops)
      input_dict = dict(input_dict.items() + curr_node_output_map)
      output_dict = dict(output_dict.items() + curr_node_output_map)
      predict_net.op.extend(output_ops)

    predict_net.output_dict = output_dict
    return original_input_dict, predict_net

  @classmethod
  def prepare(cls, model, device='CPU', **kwargs):
    super(TensorflowBackend, cls).prepare(model, device, **kwargs)

    original_input_dict, predict_net = cls.onnx_graph_to_tensorflow_net(model.graph)

    initialized = {init.name for init in model.graph.initializer}
    uninitialized = [x for x in predict_net.external_input
                     if not x in initialized]

    return TensorflowRep(predict_net, original_input_dict, uninitialized)

  @classmethod
  def onnx_initializer_to_input_dict_items(cls, initializer, init_net_name='init'):
    def tensor2list(onnx_tensor):
      # Use the onnx.numpy_helper because the data may be raw
      return onnx.numpy_helper.to_array(onnx_tensor).flatten().tolist()
    input_dict = [(tp.name, tf.constant(tensor2list(tp), shape=tp.dims)) for tp in initializer]
    return input_dict

  @classmethod
  def op_name_to_lower(cls, name):
    return re.sub('(?<!^)(?=[A-Z])', '_', name).lower()

  @classmethod
  def _onnx_node_to_tensorflow_op(cls, node, input_dict):
    op_name_lowered = cls.op_name_to_lower(node.op_type)
    if op_name_lowered in cls.onnx_tf_op_map.keys():
      return cls.handle_trivial(node, input_dict)

    handler_name = "handle_" + op_name_lowered
    # Check if specialized handler exists.
    if handler_name in dir(cls):
      method_to_call = getattr(cls, handler_name)
      return method_to_call(node, input_dict)

  @classmethod
  def handle_trivial(cls, node, input_dict):
    # Perform automatic attribute value translation.
    attrs = dict([(x, cls.attr_translator[x](cls, node.attrs[x]) \
      if x in cls.attr_translator else node.attrs[x]) \
      for x in node.attrs.keys()])

    # Create an identity map from onnx attribute names to tf
    # attribute names.
    attr_map = dict([(x, x) for x in node.attrs.keys()])

    # Modify the map accoridng to onnx_tf_attribute_map.
    attr_map = dict([(x, cls.onnx_tf_attribute_map[x] \
      if x in cls.onnx_tf_attribute_map.keys() else x) \
      for x in attr_map.keys()])

    # TODO: Per op attribute name mapping has the final say.

    # Substitute attribute names in attrs.
    attrs = dict([(attr_map[x], y) for (x, y) in attrs.items()])
    inputs = [input_dict[name] for name in node.inputs]
    return [cls.onnx_tf_op_map[cls.op_name_to_lower(node.op_type)] \
      (*inputs, **attrs)]

  @classmethod
  def handle_add(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    y = input_dict[node.inputs[1]]
    if "broadcast" in node.attrs.keys():
      broadcast = node.attrs["broadcast"]
    else:
      broadcast = 1
    if broadcast == 0:
      warnings.warn("Definition of Add with broadcast disabled is incompatible"
                    "between onnx and tensorflow.", UserWarning)
    if "axis" in node.attrs.keys():
      warnings.warn("Unsupported axis attribute by Tensorflow in Add."
                    "This attribute will be ignored.", UserWarning)
    return [tf.add(x, y)]

  @classmethod
  def handle_arg_max(cls, node, input_dict):
    data = input_dict[node.inputs[0]]
    axis = node.attrs["axis"]
    keepdims = node.attrs.get("keepdims", 1)
    if keepdims == 1:
      warnings.warn("Definition of ArgMax with keepdims enabled is "
                    "incompatible between onnx and tensorflow.",
                    UserWarning)
    return [tf.argmax(data, axis=axis)]

  @classmethod
  def handle_arg_min(cls, node, input_dict):
    data = input_dict[node.inputs[0]]
    axis = node.attrs["axis"]
    keepdims = node.attrs.get("keepdims", 1)
    if keepdims == 1:
      warnings.warn("Definition of ArgMin with keepdims enabled is "
                    "incompatible between onnx and tensorflow.",
                    UserWarning)
    return [tf.argmin(data, axis=axis)]

  @classmethod
  def _pool(cls, node, input_dict, pooling_type):
    x = input_dict[node.inputs[0]]
    x_rank = len(x.get_shape())
    data_format = "NCDHW"
    if x_rank == 3:
      data_format = "NCW"
    elif x_rank == 4:
      data_format = "NCHW"
    kernel_shape = node.attrs["kernel_shape"]
    strides = node.attrs["strides"]
    if x_rank > 5:
      warnings.warn("Unsupported tensor rank in pool operator.",
                    UserWarning)
    if "pads" in node.attrs.keys():
      warnings.warn("Unsupported pads attribute by Tensorflow in "
                    "pool operator. The SAME padding algorithm will be used.",
                    UserWarning)
    return [tf.nn.pool(x, kernel_shape, pooling_type, "SAME", strides=strides,
                       data_format=data_format)]

  @classmethod
  def handle_average_pool(cls, node, input_dict):
    return cls._pool(node, input_dict, "AVG")

  @classmethod
  def handle_batch_normalization(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    scale = input_dict[node.inputs[1]]
    bias = input_dict[node.inputs[2]]
    mean = input_dict[node.inputs[3]]
    variance = input_dict[node.inputs[4]]
    variance_epsilon = node.attrs["epsilon"]
    if "is_test" in node.attrs.keys():
      warnings.warn("Unsupported is_test attribute by Tensorflow in "
                    "batch_normalization. This attribute will be ignored.",
                    UserWarning)
    if "momentum" in node.attrs.keys():
      warnings.warn("Unsupported momentum attribute by Tensorflow in "
                    "batch_normalization. This attribute will be ignored.",
                    UserWarning)
    if "spatial" in node.attrs.keys():
      warnings.warn("Unsupported spatial attribute by Tensorflow in "
                    "batch_normalization. This attribute will be ignored.",
                    UserWarning)
    return [tf.nn.batch_normalization(x, mean, variance, bias, scale,
                                      variance_epsilon)]

  @classmethod
  def handle_concat(cls, node, input_dict):
    values = [input_dict[a] for a in node.inputs]
    axis = node.attrs["axis"]
    return [tf.concat(values, axis=axis)]

  @classmethod
  def handle_constant(cls, node, input_dict):
    value = node.attrs["value"]
    elements = onnx.numpy_helper.to_array(value).flatten().tolist()
    dtype = cls.tensor_type_to_tf_type[value.data_type]
    return [tf.constant(elements, dtype=dtype, shape=value.dims)]

  @classmethod
  def handle_div(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    y = input_dict[node.inputs[1]]
    broadcast = node.attrs["broadcast"]
    if broadcast == 0:
      warnings.warn("Definition of Div with broadcast disabled is incompatible"
        "between onnx and tensorflow.", UserWarning)
    if "axis" in node.attrs.keys():
      warnings.warn("Unsupported alpha attribute by Tensorflow in Div."
        "This attribute will be ignored.", UserWarning)
    return [tf.divide(x, y)]

  @classmethod
  def handle_dropout(cls, node, input_dict):
    x = input_dict[nodse.inputs[0]]
    # Not supported by TF
    is_test = node.attrs["is_test"] if "is_test" in node.attrs.keys() else 0
    if is_test != 0:
      warnings.warn("Unsupported is_test attribute by Tensorflow in Dropout."
        "This attribute will be ignored.", UserWarning)
    ratio = node.attrs["ratio"] if "ratio" in node.attrs.keys() else 0.5
    return [tf.nn.dropout(x, 1 - ratio)]

  @classmethod
  def handle_elu(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    if "alpha" in node.attrs.keys():
      warnings.warn("Unsupported alpha attribute by Tensorflow in Elu."
        "This attribute will be ignored.", UserWarning)
    return [tf.nn.elu(x)]

  @classmethod
  def handle_flatten(cls, node, input_dict):
    tensor = input_dict[node.inputs[0]]
    axis = node.attrs["axis"] if "axis" in node.attrs.keys() else 1
    shape = tf.shape(tensor)
    split0, split1 = tf.split(shape, [axis, tf.size(shape) - axis])
    split0 = tf.reduce_prod(split0)
    split1 = tf.reduce_prod(split1)
    output_shape = tf.stack([split0, split1])
    return [tf.reshape(tensor, output_shape)]

  @classmethod
  def handle_gemm(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    y = input_dict[node.inputs[1]]
    z = input_dict[node.inputs[2]]
    if "transA" in node.attrs.keys() and node.attrs["transA"] == 1:
      x = tf.transpose(x)
    if "transB" in node.attrs.keys() and node.attrs["transB"] == 1:
      y = tf.transpose(y)
    alpha = node.attrs["alpha"] if "alpha" in node.attrs.keys() else 1.0
    beta = node.attrs["beta"] if "beta" in node.attrs.keys() else 1.0
    return [alpha * tf.matmul(x, y) + beta * z]

  @classmethod
  def handle_global_average_pool(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    shape = tf.shape(x)
    _, window_shape = tf.split(shape, [2, tf.size(shape) - 2])
    return [tf.reduce_mean(x, axis=window_shape, keep_dims=True)]

  @classmethod
  def handle_global_max_pool(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    shape = tf.shape(x)
    _, window_shape = tf.split(shape, [2, tf.size(shape) - 2])
    return [tf.reduce_max(x, axis=window_shape, keep_dims=True)]

  @classmethod
  def handle_l_r_n(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    alpha = node.attrs["alpha"]
    beta = node.attrs["beta"]
    bias = node.attrs["bias"]
    size = node.attrs["size"]
    tf_alpha = alpha * 1.0 / size
    depth_radius = np.floor([(size - 1) / 2.0])[0]
    return [tf.nn.lrn(x, depth_radius=depth_radius,
      bias=bias, alpha=tf_alpha, beta=beta)]

  @classmethod
  def handle_leaky_relu(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    if not "alpha" in node.attrs.keys():
      warnings.warn("Provide an alpha value.", UserWarning)
      alpha = 1.0
    else:
      alpha = node.attrs["alpha"]
    tf_op = tf.nn.relu(x) - alpha * tf.nn.relu(-x)
    return [tf_op]

  @classmethod
  def handle_max(cls, node, input_dict):
    values = [input_dict[a] for a in node.inputs]
    return [tf.reduce_max(tf.stack(values), axis=0)]

  @classmethod
  def handle_max_pool(cls, node, input_dict):
    # The only two attributes that are used as is are
    #    - kernel_shape
    #    - strides
    # `dilations` is not supported by Tensorflow.
    # `pads` is replaced with Tensorflow's "SAME" padding.
    x = input_dict[node.inputs[0]]
    if "dilations" in node.attrs.keys():
      dilations = node.attrs["dilations"]
      all_ones = True
      for i in range(len(dilations)):
        if dilations[i] != 1:
          all_ones = False
      if not all_ones:
        warnings.warn("No dilations supported by Tensorflow.", UserWarning)
    kernel_shape = node.attrs["kernel_shape"]
    # TODO: map ONNX padding to TF padding. For now default to "SAME".
    pads = node.attrs["pads"]
    strides = node.attrs["strides"]
    # Also takes data_format='NHWC'
    # TF only works on 3D and 4D tensors
    if len(kernel_shape) == 4:
      return [tf.nn.max_pool(x, ksize=kernel_shape,
        strides=strides, padding="SAME")]
    elif len(kernel_shape) >= 5:
      return [tf.nn.max_pool3d(x, ksize=kernel_shape,
        strides=strides, padding="SAME")]
    else:
      # TODO: do pooling using other TF operations.
      warnings.warn("Max pool not supported for this tensor size",
        UserWarning)
      return []

  @classmethod
  def handle_min(cls, node, input_dict):
    values = [input_dict[a] for a in node.inputs]
    return [tf.reduce_min(tf.stack(values), axis=0)]

  @classmethod
  def handle_mul(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    y = input_dict[node.inputs[1]]
    if "broadcast" in node.attrs.keys():
      broadcast = node.attrs["broadcast"]
    else:
      broadcast = 1
    if broadcast == 0:
      warnings.warn("Definition of Mul with broadcast disabled is incompatible"
        "between onnx and tensorflow.", UserWarning)
    if "axis" in node.attrs.keys():
      warnings.warn("Unsupported alpha attribute by Tensorflow in Mul."
        "This attribute will be ignored.", UserWarning)
    return [tf.multiply(x, y)]

  #TODO: better support optimized rnn
  @classmethod
  def handle_optimized_r_n_n(cls, node, input_dict):
    if "direction" in node.attrs.keys():
      direction = node.attrs["direction"]
    else:
      direction = 1
    if "num_layers" in node.attrs.keys():
      num_layers = node.attrs["num_layers"]
    else:
      num_layers = 1
    if "skip_input_transform" in node.attrs.keys():
      warnings.warn("We currently do not support skipping input transformation.")

    hidden_size = node.attrs["hidden_size"]
    if node.attrs["cell_type"] == "relu":
      relu_layer = tf.contrib.rnn.BasicCell(hidden_size, activation=tf.nn.relu)
      cell = tf.contrib.rnn.MultiRNNCell([relu_layer] * num_layers)
    elif node.attrs["cell_type"] == "tanh":
      tanh_layer = tf.contrib.rnn.BasicCell(hidden_size)
      cell = tf.contrib.rnn.MultiRNNCell([tanh_layer] * num_layers)
    elif node.attrs["cell_type"] == "gru":
      gru_layer = tf.contrib.rnn.GRUCell(hidden_size)
      cell = tf.contrib.rnn.MultiRNNCell([gru_layer] * num_layers)
    elif node.attrs["cell_type"] == "lstm":
      lstm_layer = tf.contrib.rnn.LSTMCell(hidden_size)
      cell = tf.contrib.rnn.MultiRNNCell([lstm_layer] * num_layers)
    else:
      raise RuntimeError("unexpected cell type")

    warnings.warn("Initial weight, hidden/cell states will be ignored for now.")
    # TODO: handle data types
    if direction == 1:
      output, state = tf.nn.dynamic_rnn(cell,
                                        input_dict[node.inputs[1]],
                                        time_major=True,
                                        dtype=tf.float32)
    else:
      output, state = tf.nn.bidirectional_dynamic_rnn(cell,
                                                      input_dict[node.inputs[1]],
                                                      time_major=True,
                                                      dtype=tf.float32)

    if node.attrs["cell_type"] == "lstm":
      state = state[0]
      c, h = state
      states = [h, c]
    else:
      states = [state]
    outputs = [output]
    outputs.extend(states)
    return outputs

  @classmethod
  def handle_p_relu(cls, node, input_dict):
    """
    Reference implementation at
    https://github.com/tflearn/tflearn/blob/4ba8c8d78bf1bbdfc595bf547bad30580cb4c20b/tflearn/activations.py#L191
    """
    x = input_dict[node.inputs[0]]
    slope = input_dict[node.inputs[1]]
    pos = tf.nn.relu(x)
    neg = slope * (x - abs(x)) * 0.5
    return [pos + neg]

  @classmethod
  def handle_pad(cls, node, input_dict):
    mode = node.attrs["mode"]
    value = node.attrs["value"]
    num_dim = int(len(node.attrs["paddings"])/2)
    padding = tf.constant(np.array(node.attrs["paddings"])
                          .reshape([num_dim, 2])
                          .astype(np.int32)) # tf requires int32 paddings
    return [tf.pad(input_dict[node.inputs[0]], padding, mode, None, value)]

  @classmethod
  def handle_random_normal_like(cls, node, input_dict):
    shape = tf.shape(input_dict[node.inputs[0]])
    mean = node.attrs["mean"]
    stddev = node.attrs["scale"]
    dtype = cls.tensor_type_to_tf_type[node.attrs["dtype"]]
    seed = node.attrs["seed"] if "seed" in node.attrs.keys() else None
    return [tf.random_normal(shape, mean, stddev, dtype, seed)]

  @classmethod
  def handle_random_uniform_like(cls, node, input_dict):
    shape = tf.shape(input_dict[node.inputs[0]])
    minval = node.attrs["low"]
    maxval = node.attrs["high"]
    dtype = cls.tensor_type_to_tf_type[node.attrs["dtype"]]
    seed = node.attrs["seed"] if "seed" in node.attrs.keys() else None
    return [tf.random_uniform(shape, minval, maxval, dtype, seed)]

  @classmethod
  def handle_reshape(cls, node, input_dict):
    tensor = input_dict[node.inputs[0]]
    shape = tf.constant(node.attrs["shape"])
    return [tf.reshape(tensor, shape)]

  @classmethod
  def handle_selu(cls, node, input_dict):
    warnings.warn("Definition of Selu is incompatible"
      "between onnx and tensorflow.", UserWarning)
    return [tf.nn.selu(input_dict[node.inputs[0]])]

  # TODO: take care of negative indicies, discontinuous axes
  @classmethod
  def handle_slice(cls, node, input_dict):
    shape = tf.reshape(tf.rank(input_dict[node.inputs[0]]), tf.constant([1]))
    indices = tf.expand_dims(input_dict[node.inputs[1]], -1)
    begin = tf.scatter_nd(indices, input_dict[node.inputs[2]], shape)
    end = tf.scatter_nd(indices, input_dict[node.inputs[3]], shape)
    size = end - begin
    return [tf.slice(input_dict[node.inputs[0]], begin, size)]

  @classmethod
  def handle_split(cls, node, input_dict):
    split = tf.constant(node.attrs["split"]) if "split" in node.attrs else input_dict[node.inputs[1]]
    axis = node.attrs["axis"]
    return [tf.split(input_dict[node.inputs[0]], split, axis)]

  @classmethod
  def handle_sub(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    y = input_dict[node.inputs[1]]
    broadcast = node.attrs["broadcast"]
    if broadcast == 0:
      warnings.warn("Definition of Sub with broadcast disabled is incompatible"
        "between onnx and tensorflow.", UserWarning)
    if "axis" in node.attrs.keys():
      warnings.warn("Unsupported axis attribute by Tensorflow in Sub."
        "This attribute will be ignored.", UserWarning)
    return [tf.subtract(x, y)]

  @classmethod
  def handle_sum(cls, node, input_dict):
    values = [input_dict[a] for a in node.inputs]
    return [tf.reduce_sum(tf.stack(values), axis=0)]

  @classmethod
  def supports_device(cls, device):
    if device == "GPU":
      local_device_protos = device_lib.list_local_devices()
      return len([x.name for x in local_device_protos if x.device_type == 'GPU']) > 0
    elif device == "CPU":
      return True
    return False

prepare = TensorflowBackend.prepare

run_node = TensorflowBackend.run_node

run_model = TensorflowBackend.run_model

supports_device = TensorflowBackend.supports_device
