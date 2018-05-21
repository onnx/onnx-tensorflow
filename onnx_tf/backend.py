"""Backend for running ONNX on Tensorflow

To run this, you will need to have Tensorflow installed as well.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import importlib
import warnings
import sys
from math import ceil, floor

try:
  from itertools import izip as zip
except ImportError:  # will be 3.x series
  pass

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from onnx_tf.tf_net import TensorflowNet
from onnx_tf.backend_rep import TensorflowRep
from onnx_tf.opset_version import backend_opset_version
from onnx_tf.common import (ONNX_OP_TO_TF_OP, ONNX_ATTR_TO_TF_ATTR,
                            ONNX_ATTR_TO_TF_ATTR_PER_OP,
                            ONNX_ATTR_TO_REMOVE_PER_OP, ONNX_TYPE_TO_TF_TYPE,
                            TF_TYPE_ENUM, op_name_to_lower,
                            PAD_TF_INCOMPATIBLE)
from onnx.backend.base import (
    Backend,
    Device,
    DeviceType,
    namedtupledict,
)

from onnx import defs
from onnx import numpy_helper


# TODO: allow more flexible placement
def get_device_option(device):
  m = {DeviceType.CPU: '/cpu', DeviceType.CUDA: '/gpu'}
  return m[device.type]


# TODO: Move this into ONNX main library
def convert_attribute_proto(onnx_arg):
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
      d[arg.name] = convert_attribute_proto(arg)
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


class TensorflowBackendBase(Backend):
  """ Tensorflow Backend for ONNX
  """
  attr_translator = {
      "dtype": lambda cls, x: ONNX_TYPE_TO_TF_TYPE[x],
      "keepdims": lambda cls, x: bool(x),
      "to": lambda cls, x: ONNX_TYPE_TO_TF_TYPE[x],
  }

  DEFAULT_ONNX_ATTR_PER_OP = {
      "random_normal": {
          "mean": 0,
          "scale": 1
      },
      "random_uniform": {
          "low": 0,
          "high": 1
      },
      "reduce_log_sum_exp": {
          "keepdims": 1
      },
      "reduce_max": {
          "keepdims": 1
      },
      "reduce_mean": {
          "keepdims": 1
      },
      "reduce_min": {
          "keepdims": 1
      },
      "reduce_prod": {
          "keepdims": 1
      },
      "reduce_sum": {
          "keepdims": 1
      },
      "shape": {
          "out_type": tf.int64
      },
      "size": {
          "out_type": tf.int64
      },

      # Force to use NCHW temporally
      # https://github.com/onnx/onnx/pull/443
      "conv": {
          "data_format": "channels_first"
      },
      "max_pool": {
          "data_format": "NCHW"
      },
      "average_pool": {
          "data_format": "NCHW"
      },
  }

  backend_version_cache = {}

  # input_shape, kernel_shape, strides are specified for
  # spatial dims only.
  @classmethod
  def get_tf_pad(cls, input_shape, kernel_shape, strides, pads):
    assert pads is not None
    num_dim = int(len(input_shape))
    num_sp_dim = int(len(kernel_shape))

    if pads == [0] * num_sp_dim * 2 or pads is None:
      return "VALID"

    is_same_padding = True
    for (input_size, stride_size, kernel_size, left_pad, right_pad) in zip(
        input_shape, strides, kernel_shape, pads[:num_sp_dim],
        pads[num_sp_dim:]):
      output_size = ceil(float(input_size) / float(stride_size))
      padding_total = int(
          (output_size - 1) * stride_size + kernel_size - input_size)
      padding_left = int(floor(float(padding_total) / 2.0))
      padding_right = padding_total - padding_left

      is_same_padding = is_same_padding and (left_pad == padding_left and
                                             right_pad == padding_right)

    if is_same_padding:
      return "SAME"

    return PAD_TF_INCOMPATIBLE

  @classmethod
  def get_padding_as_op(cls, x, pads):
    num_dim = int(len(pads) / 2)

    tf_pads = np.transpose(np.array(pads).reshape([2, num_dim]))
    tf_pads = [0, 0, 0, 0] + tf_pads.flatten().tolist()

    padding = tf.constant(
        np.array(tf_pads).reshape([num_dim + 2, 2])
        .astype(np.int32))  # tf requires int32 paddings
    return tf.pad(x, padding)

  @classmethod
  def _explicit_broadcast(cls, tensor, broadcast_dim=1, total_num_dim=4):
    if broadcast_dim < 0:
      broadcast_dim += total_num_dim
    dims = [broadcast_dim + i for i in range(len(tensor.shape))]
    for i in range(total_num_dim):
      if i not in dims:
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
      y = cls._explicit_broadcast(y, node.attrs["axis"], len(x.shape))

    return op_func(x, y)

  @classmethod
  def run_node(cls, node, inputs, device='CPU'):
    super(TensorflowBackendBase, cls).run_node(node, inputs, device)
    node_graph = tf.Graph()
    with node_graph.as_default():
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
      input_dict = dict(
          [(x[0], tf.constant(x[1])) for x in feed_dict_raw.items()])
      ops = cls._onnx_node_to_tensorflow_op(node, input_dict)
      output_vals = []

      with tf.Session() as sess:
        with tf.device(device_option):
          sess.run(tf.global_variables_initializer())
          output_vals = sess.run(ops)

    return namedtupledict('Outputs', node.outputs)(*output_vals)

  @classmethod
  def onnx_graph_to_tensorflow_net(cls, graph_def, opset):
    model_graph = tf.Graph()
    with model_graph.as_default():
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
      predict_net.graph = model_graph

      predict_net.external_input.extend(value_info.name
                                        for value_info in graph_def.input
                                        if value_info.name not in initialized)

      predict_net.external_output.extend(
          value_info.name for value_info in graph_def.output)

      # creating placeholders for currently unkown inputs
      for value_info in graph_def.input:
        if value_info.name in initialized:
          continue
        shape = list(
            d.dim_value if (d.dim_value >= 0 and d.dim_param == "") else None
            for d in value_info.type.tensor_type.shape.dim)
        x = tf.placeholder(
            TF_TYPE_ENUM[value_info.type.tensor_type.elem_type],
            name=value_info.name,
            shape=shape)
        input_dict_items.append([value_info.name, x])

      # tensor dict: this dictionary is a map from variable names
      # to the latest produced TF tensors of the given name.
      # This dictionary will get updated as we build the graph to
      # record the names of newly produced tensors.
      tensor_dict = dict(input_dict_items)
      # Since tensor dict may be updated, we need to keep a copy
      # of the original input dict where we track the earliest
      # defined tensors so we can have access to the placeholders
      # to feed in input tensors when we run the graph.
      input_dict = dict(input_dict_items)

      for node in graph_def.node:
        node = OnnxNode(node)

        output_ops = cls._onnx_node_to_tensorflow_op(
            node, tensor_dict, opset=opset)
        curr_node_output_map = list(zip(node.outputs, output_ops))
        tensor_dict = dict(list(tensor_dict.items()) + curr_node_output_map)

      predict_net.tensor_dict = tensor_dict

    return predict_net

  @classmethod
  def prepare(cls, model, device='CPU', **kwargs):
    """Prepare an ONNX model for Tensorflow Backend

    This function converts an ONNX model to an internel representation
    of the computational graph called TensorflowRep and returns
    the converted representation.

    :param model: the ONNX model to be converted
    :param device: the device to execute this model on

    :returns: a TensorflowRep class object representing the ONNX model
    """
    super(TensorflowBackendBase, cls).prepare(model, device, **kwargs)

    predict_net = (cls.onnx_graph_to_tensorflow_net(
        model.graph, opset=model.opset_import[0].version))

    return TensorflowRep(predict_net)

  @classmethod
  def onnx_initializer_to_input_dict_items(cls,
                                           initializer,
                                           init_net_name='init'):

    def tensor2list(onnx_tensor):
      # Use the onnx.numpy_helper because the data may be raw
      return numpy_helper.to_array(onnx_tensor).flatten().tolist()

    input_dict = [(tp.name,
                   tf.constant(
                       tensor2list(tp),
                       shape=tp.dims,
                       dtype=ONNX_TYPE_TO_TF_TYPE[tp.data_type]))
                  for tp in initializer]
    return input_dict

  @classmethod
  def _onnx_node_to_tensorflow_op(cls, node, input_dict, opset=0):
    """
    Convert onnx node to tensorflow op.

    Args:
      node: Onnx node object.
      input_dict: Inputs dict of graph.
      opset: Opset version of the operator set. Default 0 means using latest version.

    Returns:
      Tensorflow op
    """
    op_name_lowered = op_name_to_lower(node.op_type)
    handler_name = "handle_" + op_name_lowered

    # Check if specialized handler exists.
    versions = backend_opset_version[node.op_type]

    if opset == 0:
      # use the maximum opset version available that is
      # smaller or equal to the version supported by
      # the onnx package.
      versions = filter(lambda v: v <= defs.onnx_opset_version(), versions)
      version = max(versions)
    else:
      versions = sorted(versions + [opset])
      version = versions[max([i for i, v in enumerate(versions) if v == opset])
                         - 1]

    backend_ver = 'backend_v{}'.format(version)
    backend = cls.backend_version_cache.setdefault(
        backend_ver,
        importlib.import_module(
            'onnx_tf.backends.' + backend_ver).TensorflowBackend)

    if hasattr(backend, handler_name):
      method_to_call = getattr(backend, handler_name)
      return method_to_call(node, input_dict)
    elif op_name_lowered in ONNX_OP_TO_TF_OP.keys():
      return backend.handle_trivial(node, input_dict)
    else:
      raise NotImplementedError("{} op is not implemented.".format(
          node.op_type))

  @classmethod
  def handle_trivial(cls, node, input_dict):
    op_name_lowered = op_name_to_lower(node.op_type)

    attrs = dict([(x, node.attrs[x]) for x in node.attrs.keys()])

    if op_name_lowered in cls.DEFAULT_ONNX_ATTR_PER_OP:
      default_attrs = cls.DEFAULT_ONNX_ATTR_PER_OP[op_name_lowered]
      default_attrs.update(attrs)
      attrs = default_attrs

    # Perform automatic attribute value translation.
    attrs = dict([(x, cls.attr_translator[x](cls, attrs[x]) \
      if x in cls.attr_translator else attrs[x]) \
                  for x in attrs.keys()])

    # Create an identity map from onnx attribute names to tf
    # attribute names.
    attr_map = dict([(x, x) for x in attrs.keys()])

    # Modify the map accoridng to onnx_tf_attribute_map.
    attr_map = dict([(x, ONNX_ATTR_TO_TF_ATTR[x] \
      if x in ONNX_ATTR_TO_TF_ATTR.keys() else x) \
                     for x in attr_map.keys()])

    # TODO: Per op attribute name mapping has the final say.

    # Modify the map according to onnx_tf_per_op_attr_map
    attr_map = dict([(x, ONNX_ATTR_TO_TF_ATTR_PER_OP[op_name_lowered][x]
                      if op_name_lowered in ONNX_ATTR_TO_TF_ATTR_PER_OP and
                      x in ONNX_ATTR_TO_TF_ATTR_PER_OP[op_name_lowered].keys()
                      else attr_map[x]) for x in attr_map.keys()])

    # Substitute attribute names in attrs.
    attrs = dict([(attr_map[x], y) for (x, y) in attrs.items()])
    # Remove the key according to onnx_tf_per_op_attr_remove
    attrs = {
        x: attrs[x]
        for x in attrs
        if not (op_name_lowered in ONNX_ATTR_TO_REMOVE_PER_OP and
                x in ONNX_ATTR_TO_REMOVE_PER_OP[op_name_lowered])
    }
    inputs = [input_dict[name] for name in node.inputs]
    return [ONNX_OP_TO_TF_OP[op_name_to_lower(node.op_type)] \
              (*inputs, **attrs)]

  @classmethod
  def get_data_format(cls, x_rank, support_cuda):
    sp_dim_names = ["D", "H", "W"]
    sp_dim_lst = []
    for i in range(x_rank - 2):
      sp_dim_lst.append(sp_dim_names[-i - 1])

    sp_dim_string = "".join(reversed(sp_dim_lst))
    storage_format = "NC" + sp_dim_string

    if support_cuda:
      compute_format = "NC" + sp_dim_string
    else:
      compute_format = "N" + sp_dim_string + "C"
    return storage_format, compute_format

  @classmethod
  def supports_device(cls, device):
    if device == "CUDA":
      local_device_protos = device_lib.list_local_devices()
      return len(
          [x.name for x in local_device_protos if x.device_type == 'GPU']) > 0
    elif device == "CPU":
      return True
    return False


prepare = TensorflowBackendBase.prepare

run_node = TensorflowBackendBase.run_node

run_model = TensorflowBackendBase.run_model

supports_device = TensorflowBackendBase.supports_device
