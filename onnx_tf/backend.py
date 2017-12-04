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
import sys
import itertools
from math import ceil, floor

try:
  from itertools import izip as zip
except ImportError: # will be 3.x series
  pass

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
    return str(onnx_arg.s, 'utf-8') \
      if sys.version_info[0] >= 3 else onnx_arg.s
  elif onnx_arg.HasField('t'):
    return onnx_arg.t  # this is a proto!
  elif onnx_arg.floats:
    return list(onnx_arg.floats)
  elif onnx_arg.ints:
    return list(onnx_arg.ints)
  elif onnx_arg.strings:
    str_list = list(onnx_arg.strings)
    if sys.version_info[0] >= 3:
      str_list = map(lambda x: str(x, 'utf-8'), str_list)
    return str_list
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
    self.node_proto = node

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
      "softplus": tf.nn.softplus,
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
      tf.string,
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

  # input_shape, kernel_shape, strides are specified for
  # spatial dims only.
  @classmethod
  def get_tf_pad(cls, input_shape, kernel_shape, strides, pads):
    num_dim = int(len(input_shape))
    num_sp_dim = int(len(kernel_shape))

    if pads == [0] * num_sp_dim * 2 or pads == None:
      return "VALID"

    is_same_padding = True
    for (input_size,
         stride_size,
         kernel_size,
         left_pad,
         right_pad) in zip(input_shape,
                           strides,
                           kernel_shape,
                           pads[:num_sp_dim],
                           pads[num_sp_dim:]):

      output_size = ceil(float(input_size) / float(stride_size))
      padding_total = int((output_size - 1) * stride_size +
                          kernel_size - input_size)
      padding_left = int(floor(float(padding_total) / 2.0))
      padding_right = padding_total - padding_left

      is_same_padding = is_same_padding and (left_pad == padding_left and
                                             right_pad == padding_right)


    if is_same_padding:
      return "SAME"

    return None

  @classmethod
  def get_padding_as_op(cls, x, pads):
    num_dim = int(len(pads)/2)

    tf_pads = np.transpose(np.array(pads).reshape([2, num_dim]))
    tf_pads = [0, 0, 0, 0] + tf_pads.flatten().tolist()

    padding = tf.constant(np.array(tf_pads)
                          .reshape([num_dim + 2, 2])
                          .astype(np.int32)) # tf requires int32 paddings
    return tf.pad(x, padding)

  @classmethod
  def _explicit_broadcast(cls, tensor, broadcast_dim=1, total_num_dim=4):
    if not isinstance(broadcast_dim, list):
      broadcast_dim = [broadcast_dim]

    for i in range(total_num_dim):
      if i not in broadcast_dim:
        tensor = tf.expand_dims(tensor, i)

    return tensor

  @classmethod
  def _bin_op(cls, node, input_dict, op_func):
    x = input_dict[node.inputs[0]]
    y = input_dict[node.inputs[1]]
    broadcast = node.attrs.get("broadcast", 1)
    if broadcast == 0:
      warnings.warn("Definition of {} with broadcast disabled is not "
                    "yet supported.".format(node.type), UserWarning)

    if "axis" in node.attrs.keys():
      num_ones_to_append = len(x.get_shape()) - \
                           len(y.get_shape()) - \
                           node.attrs["axis"]
      if num_ones_to_append > 0:
        ones = tf.ones([num_ones_to_append], tf.int32)
        broadcasted_shape = tf.concat([tf.shape(y), ones], axis=0)
        y = tf.reshape(y, broadcasted_shape)

    return op_func(x, y)

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
        sess.run(tf.global_variables_initializer())
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

      shape = list(d.dim_value for d in
                   value_info.type.tensor_type.shape.dim)
      x = tf.placeholder(cls.tensor_type_enum[
          value_info.type.tensor_type.elem_type],
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
      curr_node_output_map = list(zip(node.outputs, output_ops))
      input_dict = dict(list(input_dict.items()) +
                        curr_node_output_map)

      output_dict = dict(list(output_dict.items()) +
                         curr_node_output_map)
      predict_net.op.extend(output_ops)
    predict_net.output_dict = output_dict
    return original_input_dict, predict_net

  @classmethod
  def prepare(cls, model, device='CPU', **kwargs):
    super(TensorflowBackend, cls).prepare(model, device, **kwargs)

    original_input_dict, predict_net = (
        cls.onnx_graph_to_tensorflow_net(model.graph))

    initialized = {init.name for init in model.graph.initializer}
    uninitialized = [x for x in predict_net.external_input
                     if not x in initialized]

    return TensorflowRep(predict_net, original_input_dict, uninitialized)

  @classmethod
  def onnx_initializer_to_input_dict_items(cls,
                                           initializer,
                                           init_net_name='init'):
    def tensor2list(onnx_tensor):
      # Use the onnx.numpy_helper because the data may be raw
      return onnx.numpy_helper.to_array(onnx_tensor).flatten().tolist()
    input_dict = [(tp.name, tf.constant(tensor2list(tp),
                                        shape=tp.dims,
                                        dtype=cls.tensor_type_to_tf_type[
                                            tp.data_type]))
                  for tp in initializer]
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
    else:
      raise NotImplementedError("{} op is not implemented.".format(node.op_type))

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
    return [cls._bin_op(node, input_dict, tf.add)]

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
  def _compatibility_avg_pool(cls,
                              node,
                              input_dict,
                              pool_func,
                              guess_or_manual_pad):
    from math import ceil

    x = input_dict[node.inputs[0]]
    x_rank = len(x.get_shape())

    kernel_shape = node.attrs["kernel_shape"]
    strides = node.attrs["strides"]

    pads = node.attrs.get("pads", [0, 0, 0, 0])

    # pylint: disable=line-too-long
    def py_pool(x, kernel_shape, strides, pad):
      out_h = int((x.shape[2] + pads[0] + pads[2] - kernel_shape[0]) // strides[0]) + 1
      out_w = int((x.shape[3] + pads[1] + pads[3] - kernel_shape[1]) // strides[1]) + 1

      out = np.zeros([x.shape[0], x.shape[1], out_h, out_w], dtype=np.float32)
      for n in range(0, x.shape[0]):
        for c in range(0, x.shape[1]):
          for h in range(0 - pad[0], x.shape[2] + pad[2], strides[0]):
            for w in range(0 - pad[1], x.shape[3] + pad[3], strides[1]):
              # skip window if window is outside padded region
              if (h + kernel_shape[0] > x.shape[2] + pad[2]) or \
                 (w + kernel_shape[1] > x.shape[3] + pad[3]):
                continue
              count = 0
              val = 0
              for kh in range(0, kernel_shape[0]):
                for kw in range(0, kernel_shape[1]):
                  current_h = h+kh
                  current_w = w+kw
                  if (current_h >= 0) and (current_w >= 0) and \
                     (current_h < x.shape[2]) and (current_w < x.shape[3]):
                    count += 1
                    val += x[n][c][current_h][current_w]
              out[n][c][int((h + pad[0])//strides[0])][int((w + pad[1])//strides[1])] = val/count
      return out

    pooled = tf.py_func(py_pool, [x, kernel_shape, strides, pads], tf.float32)
    x_shape = list(x.get_shape())

    out_h = int((x_shape[2] + pads[0] + pads[2] - kernel_shape[0]) // strides[0]) + 1
    out_w = int((x_shape[3] + pads[1] + pads[3] - kernel_shape[1]) // strides[1]) + 1
    pooled.set_shape([x_shape[0], x_shape[1], out_h, out_w])
    # pylint: enable=line-too-long

    return [pooled]

  @classmethod
  def _pool(cls, node, input_dict, pool_func, can_pad_zero):
    from math import ceil

    x = input_dict[node.inputs[0]]
    x_rank = len(x.get_shape())

    support_cuda = cls.supports_device("CUDA")
    storage_format, compute_format = cls.get_data_format(x_rank, support_cuda)

    kernel_shape = node.attrs["kernel_shape"]
    strides = node.attrs["strides"]

    # By default, do not pad
    pad = "VALID"

    if "pads" in node.attrs.keys():
      pad = cls.get_tf_pad(x.get_shape().as_list(),
                           kernel_shape,
                           strides,
                           node.attrs["pads"])
      if pad is None and can_pad_zero:
        x = cls.get_padding_as_op(x, node.attrs["pads"])
        pad = "VALID"
      if pad is None and not can_pad_zero:
        # Currently it's always average pooling
        return cls._compatibility_avg_pool(node, input_dict, tf.nn.avg_pool, 0)

    if support_cuda:
      pooled = pool_func(x, [1, 1] + kernel_shape, [1, 1] + strides, pad,
                         data_format=compute_format)
    else:
      x = tf.transpose(x, perm=[0, 2, 3, 1])
      pooled = pool_func(x, [1] + kernel_shape + [1], [1] + strides + [1], pad,
                         data_format=compute_format)
      pooled = tf.transpose(pooled, perm=[0, 3, 1, 2])

    return [pooled]

  @classmethod
  def handle_average_pool(cls, node, input_dict):
    spatial_dim = list(input_dict[node.inputs[0]].get_shape()[2:])
    kernel_shape = node.attrs.get("kernel_shape", [])
    global_pool = True
    for i in range(len(spatial_dim)):
      global_pool = global_pool and (spatial_dim[i] < kernel_shape[i])

    if global_pool:
      return cls.handle_global_average_pool(node, input_dict)

    # 0 = cannot pad zero
    return cls._pool(node, input_dict, tf.nn.avg_pool, 0)

  @classmethod
  def handle_batch_normalization(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    total_num_dim = len(x.get_shape())
    scale = cls._explicit_broadcast(input_dict[node.inputs[1]], 1, total_num_dim)
    bias = cls._explicit_broadcast(input_dict[node.inputs[2]], 1, total_num_dim)
    mean = cls._explicit_broadcast(input_dict[node.inputs[3]], 1, total_num_dim)
    variance = cls._explicit_broadcast(input_dict[node.inputs[4]], 1, total_num_dim)

    variance_epsilon = node.attrs.get("epsilon", 0.00001)
    if node.attrs.get("is_test", 0):
      return [tf.nn.batch_normalization(x, mean, variance, bias, scale,
                                        variance_epsilon)]
    if "momentum" in node.attrs.keys():
      warnings.warn("Unsupported momentum attribute by Tensorflow in "
                    "batch_normalization. This attribute will be ignored.",
                    UserWarning)
    if "spatial" in node.attrs.keys():
      warnings.warn("Unsupported spatial attribute by Tensorflow in "
                    "batch_normalization. This attribute will be ignored.",
                    UserWarning)
    # TODO: need to conform to the documentation here
    return [tf.nn.batch_normalization(x, mean, variance, bias, scale,
                                      variance_epsilon)]
  @classmethod
  def handle_clip(cls, node, input_dict):
    assert "max" in node.attrs.keys()
    assert "min" in node.attrs.keys()

    max_val = node.attrs["max"]
    min_val = node.attrs["min"]

    return [tf.clip_by_value(input_dict[node.inputs[0]], min_val, max_val)]

  @classmethod
  def handle_concat(cls, node, input_dict):
    values = [input_dict[a] for a in node.inputs]
    # apparently this is what's needed for squeezenet to work
    axis = node.attrs.get("axis", 1)
    return [tf.concat(values, axis=axis)]

  @classmethod
  def handle_constant(cls, node, input_dict):
    value = node.attrs["value"]
    elements = onnx.numpy_helper.to_array(value).flatten().tolist()
    dtype = cls.tensor_type_to_tf_type[value.data_type]
    return [tf.constant(elements, dtype=dtype, shape=value.dims)]

  @classmethod
  def get_data_format(cls, x_rank, support_cuda):
    sp_dim_names = ["D", "H", "W"]
    sp_dim_lst = []
    for i in range(x_rank-2):
      sp_dim_lst.append(sp_dim_names[-i-1])

    sp_dim_string = "".join(reversed(sp_dim_lst))
    storage_format = "NC" + sp_dim_string

    if support_cuda:
      compute_format = "NC" + sp_dim_string
    else:
      compute_format = "N" + sp_dim_string + "C"
    return storage_format, compute_format


  @classmethod
  def get_perm_from_formats(cls, _from, _to):
    return list(map(lambda x: _from.find(x), _to))

  @classmethod
  def _conv(cls, node, input_dict, transpose=False):
    x = input_dict[node.inputs[0]]
    x_rank = len(x.get_shape())

    support_cuda = cls.supports_device("CUDA")
    storage_format, compute_format = cls.get_data_format(x_rank, support_cuda)

    in_weights = input_dict[node.inputs[1]]
    weights_rank = len(in_weights.get_shape())
    if transpose:
      # Translate weights from (C x M x KH x KW) to (KH x KW X C X M)
      perm = list(range(2, weights_rank)) + [0, 1]
    else:
      # Translate weights from (M x C x KH x KW) to (KH x KW X C X M)
      perm = list(range(2, weights_rank)) + [1, 0]

    weights = tf.transpose(in_weights, perm)
    dilations = node.attrs.get("dilations", None)
    strides = node.attrs.get("strides", None)

    if "kernel_shape" in node.attrs.keys():
      warnings.warn("Unsupported kernel_shape attribute by Tensorflow in "
                    "Conv operator. The attribute will be ignored.",
                    UserWarning)

    if "pads" in node.attrs.keys():
      x = cls.get_padding_as_op(x, node.attrs["pads"])

    if "group" in node.attrs:

      weight_groups = tf.split(weights,
                               num_or_size_splits=node.attrs["group"],
                               axis=-1)

      if support_cuda:
        xs = tf.split(x, num_or_size_splits=node.attrs["group"], axis=1)
      else:
        x = tf.transpose(x, perm=cls.get_perm_from_formats(storage_format, compute_format))
        xs = tf.split(x, num_or_size_splits=node.attrs["group"], axis=-1)

      convolved = [tf.nn.convolution(x, weight, "VALID", strides=strides,
                                     dilation_rate=dilations,
                                     data_format=compute_format) for
                   (x, weight) in zip(xs, weight_groups)]

      if len(node.inputs) == 2:
        if support_cuda:
          output = tf.concat(convolved, axis=1)
        else:
          output = tf.concat(convolved, axis=-1)
          output = tf.transpose(output, perm=cls.get_perm_from_formats(compute_format, storage_format))
      else:
        bias = input_dict[node.inputs[2]]

        if support_cuda:
          output = tf.concat(convolved, axis=1)
          output = tf.nn.bias_add(output, bias, data_format=compute_format)
        else:
          output = tf.concat(convolved, axis=-1)
          output = tf.nn.bias_add(output, bias, data_format=compute_format)
          output = tf.transpose(output, perm=cls.get_perm_from_formats(compute_format, storage_format))

      return [output]

    if not support_cuda:
      x = tf.transpose(x, perm=cls.get_perm_from_formats(storage_format, compute_format))

    convolved = tf.nn.convolution(x, weights, "VALID", strides=strides,
                                  dilation_rate=dilations,
                                  data_format=compute_format)

    if not support_cuda:
      convolved = tf.transpose(convolved, perm=cls.get_perm_from_formats(compute_format, storage_format))

    if len(node.inputs) == 2:
      return [convolved]
    else:
      bias = input_dict[node.inputs[2]]
      if not support_cuda:
        convolved = tf.transpose(convolved, perm=cls.get_perm_from_formats(storage_format, compute_format))
      output = tf.nn.bias_add(convolved, bias, data_format=compute_format)
      if not support_cuda:
        output = tf.transpose(output, perm=cls.get_perm_from_formats(compute_format, storage_format))
      return [output]

  @classmethod
  def handle_conv(cls, node, input_dict):
    return cls._conv(node, input_dict)

  @classmethod
  def handle_conv_transpose(cls, node, input_dict):
    return cls._conv(node, input_dict, transpose=True)

  @classmethod
  def handle_div(cls, node, input_dict):
    return [cls._bin_op(node, input_dict, tf.divide)]

  @classmethod
  def handle_dropout(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    # Not supported by TF
    is_test = node.attrs["is_test"] if "is_test" in node.attrs.keys() else 0
    if is_test:
      return [x]
    ratio = node.attrs["ratio"] if "ratio" in node.attrs.keys() else 0.5
    return [tf.nn.dropout(x, 1 - ratio)]

  @classmethod
  def handle_elu(cls, node, input_dict):
    x = input_dict[node.inputs[0]]

    alpha = node.attrs.get("alpha", 1.0)
    if "alpha" in node.attrs.keys():
      return [tf.cast(x < 0.0, tf.float32) * alpha * (tf.exp(x) - 1.0) + tf.cast(x >= 0.0, tf.float32) * x]
    else:
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
    x = tf.contrib.layers.flatten(x)
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
    dims = tf.range(tf.rank(x))
    _, dim_window = tf.split(dims, [2, tf.size(dims) - 2])
    return [tf.reduce_mean(x, axis=dim_window, keep_dims=True)]

  @classmethod
  def handle_global_max_pool(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    dims = tf.range(tf.rank(x))
    _, dim_window = tf.split(dims, [2, tf.size(dims) - 2])
    return [tf.reduce_max(x, axis=dim_window, keep_dims=True)]

  @classmethod
  def handle_l_r_n(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    alpha = node.attrs["alpha"]
    beta = node.attrs["beta"]
    bias = node.attrs["bias"]
    size = node.attrs["size"]
    tf_alpha = alpha / size
    depth_radius = np.floor([(size - 1) / 2.0])[0]
    # TODO: LRN in tf accepts radius
    # but in ONNX/Caffe accepts diameter.
    # This could be a problem.
    x_t = tf.transpose(x, perm=[0, 2, 3, 1])
    normed = tf.nn.lrn(x_t, depth_radius=depth_radius,
                       bias=bias, alpha=tf_alpha, beta=beta)
    normed = tf.transpose(normed, perm=[0, 3, 1, 2])
    return [normed]

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
    # 1 = can pad zero
    return cls._pool(node, input_dict, tf.nn.max_pool, 1)

  @classmethod
  def handle_min(cls, node, input_dict):
    values = [input_dict[a] for a in node.inputs]
    return [tf.reduce_min(tf.stack(values), axis=0)]

  @classmethod
  def handle_mul(cls, node, input_dict):
    return [cls._bin_op(node, input_dict, tf.multiply)]

  @classmethod
  def handle_or(cls, node, input_dict):
    return [cls._bin_op(node, input_dict, tf.logical_or)]

  @classmethod
  def handle_p_relu(cls, node, input_dict):
    """
    Reference implementation at
    https://github.com/tflearn/tflearn/blob/4ba8c8d78bf1bbdfc595bf547bad30580cb4c20b/tflearn/activations.py#L191
    """
    x = input_dict[node.inputs[0]]
    slope = input_dict[node.inputs[1]]
    slope = cls._explicit_broadcast(slope, 1, len(x.get_shape()))
    pos = tf.nn.relu(x)
    neg = slope * (x - abs(x)) * 0.5
    return [pos + neg]

  @classmethod
  def handle_pad(cls, node, input_dict):
    num_dim = int(len(node.attrs["pads"])/2)
    mode = node.attrs["mode"]

    def _compatibility_edge_pad(x, pads):
      x = np.pad(x, pads, mode="edge")
      return x

    value = node.attrs.get("value", 0)
    # tf requires int32 paddings
    pads = tf.constant(np.transpose(np.array(node.attrs["pads"])
                                      .reshape([2, num_dim])
                                      .astype(np.int32)))

    x = input_dict[node.inputs[0]]
    if mode.lower() == "edge":
      return [tf.py_func(_compatibility_edge_pad, [x, pads], x.dtype)]

    return [tf.pad(input_dict[node.inputs[0]],
                   pads,
                   mode,
                   None,
                   value)]

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
    warnings.warn("Definition of Selu is different "
                  "between onnx and tensorflow.", UserWarning)
    return [tf.nn.selu(input_dict[node.inputs[0]])]

  @classmethod
  def handle_slice(cls, node, input_dict):
    x = input_dict[node.inputs[0]]

    full_sizes = x.get_shape().as_list()
    full_begin = [0] * len(full_sizes)

    starts = node.attrs.get("starts")
    ends = node.attrs.get("ends")
    slice_len = len(starts)
    axes = node.attrs.get("axes", list(range(slice_len)))

    for i in range(slice_len):
      ends[i] = full_sizes[axes[i]] + ends[i] \
                if ends[i] < 0 else ends[i]
      full_sizes[axes[i]] = ends[i] - starts[i]
      full_begin[axes[i]] = starts[i]

    return [tf.slice(input_dict[node.inputs[0]],
                     tf.constant(full_begin),
                     tf.constant(full_sizes))]

  @classmethod
  def handle_softmax(cls, node, input_dict):
    if "axis" in node.attrs:
      axis = node.attrs["axis"]
      axis = (axis if axis > 0
              else len(input_dict[node.inputs[0]].get_shape()) + axis)
    else:
      axis = 1

    return [tf.nn.softmax(input_dict[node.inputs[0]], dim=axis)]

  @classmethod
  def handle_split(cls, node, input_dict):
    split = (tf.constant(node.attrs["split"]) if
             "split" in node.attrs else input_dict[node.inputs[1]])
    axis = node.attrs["axis"]
    return [tf.split(input_dict[node.inputs[0]], split, axis)]

  @classmethod
  def handle_sub(cls, node, input_dict):
    return [cls._bin_op(node, input_dict, tf.subtract)]

  @classmethod
  def handle_sum(cls, node, input_dict):
    values = [input_dict[a] for a in node.inputs]
    return [tf.reduce_sum(tf.stack(values), axis=0)]

  @classmethod
  def handle_mat_mul(cls, node, input_dict):
    return [tf.matmul(input_dict[node.inputs[0]],
                      input_dict[node.inputs[1]])]

  @classmethod
  def supports_device(cls, device):
    if device == "CUDA":
      local_device_protos = device_lib.list_local_devices()
      return len([x.name for x in
                  local_device_protos if x.device_type == 'GPU']) > 0
    elif device == "CPU":
      return True
    return False

prepare = TensorflowBackend.prepare

run_node = TensorflowBackend.run_node

run_model = TensorflowBackend.run_model

supports_device = TensorflowBackend.supports_device
