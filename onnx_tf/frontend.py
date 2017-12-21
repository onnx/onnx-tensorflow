"""Frontend for exporting Tensorflow graph to ONNX graph

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np
from onnx_tf.common import (
  TF_TYPE_TO_ONNX_TYPE,
  TF_OP_STR_TO_ONNX_OP,
  TF_ATTR_TO_ONNX_ATTR,
  get_tf_shape_as_list,
  op_name_to_lower,
)
from onnx import onnx_pb2, helper
from onnx.helper import (
  make_tensor_value_info,
  make_tensor,
  make_graph,
  make_node,
)
from onnx.onnx_pb2 import GraphProto, TensorProto, AttributeProto
from tensorflow.python.framework.tensor_util import MakeNdarray

def get_nodes_by_name(nodes, names):
  ret = []
  for node in nodes:
    if node.name in names:
      ret.append(TensorflowNode(node))
  return ret

class TensorflowNode(object):

  # Keyed by old attribute names.
  attr_translator = {
    "_output_shapes": lambda self, x: list(map(lambda shape: get_tf_shape_as_list(shape.dim), x.list.shape)),
    "shape": lambda self, x: get_tf_shape_as_list(x.shape.dim),
    "T": lambda self, x: self.type_converter(x),
    "dtype": lambda self, x: self.type_converter(x),
    "value": lambda self, x: MakeNdarray(x.tensor),
    "seed2": lambda self, x: float(x.i),
    "seed": lambda self, x: float(x.i),
    "keep_dims": lambda self, x: int(x.b),
    "squeeze_dims": lambda self, x: list(x.list.i),
  }

  def __init__(self, node_proto):
    # storing a referece to the original protobuf object
    self.node_proto = node_proto
    self.name = node_proto.name
    self.op = node_proto.op
    self.inputs = list(node_proto.input)
    self.attr = {}
    for key, val in node_proto.attr.items():
      new_key = key

      if key in TF_ATTR_TO_ONNX_ATTR.keys():
        new_key = TF_ATTR_TO_ONNX_ATTR[key]

      if key in self.attr_translator.keys():
        self.attr[new_key] = self.attr_translator[key](self, val)
      else:
        self.attr[new_key] = val

  def type_converter(self, x):
    return TF_TYPE_TO_ONNX_TYPE[tf.as_dtype(x.type)]

class TensorflowFrontend(object):
  """ Tensorflow Frontend for ONNX
  """

  @classmethod
  def tensorflow_graph_to_onnx_graph(cls, graph_def, output, device='CPU', channel_last=True, name="graph"):
    """Function that converts a tensorflow graph to an onnx graph.

    Args:
        graph_def: Tensorflow Graph Proto object.
        output: A Tensorflow NodeDef object specifying which node
          to be taken as output of the ONNX graph.
        name: The name of the output ONNX Graph.

    Returns:
        The equivalent ONNX Graph Proto object.

    """
    cls._channel_last = channel_last
    cls._device = device

    # This list holds the protobuf objects of type ValueInfoProto
    # representing the input to the converted ONNX graph.
    inputs_proto = []

    # This list holds the protobuf objects of type NodeProto
    # representing the ops in the converted ONNX graph.
    ops_proto = []

    # This dictionary contains a map from the name of the constant
    # op to the array of values it holds. This is useful because
    # tensorflow is less eager to know about input values at
    # graph construction time than ONNX. That is to say, some ONNX
    # attributes are input tensors in TF. This dictionary extracts
    # those values of constant tensors that are known at graph
    # construction time.
    consts = {}

    # Sometimes the constants are used as inputs to ops. This list
    # holds initializers that creates global constant tensors available
    # to be accessed by ops as inputs (as oppose to attributes which
    # is supplied by the `consts` map above).
    consts_proto = []

    for node in graph_def.node:
      node = TensorflowNode(node)
      if node.op == "Placeholder":
        # Tensorflow requires dtype to be known.
        # TODO: currently `dtype` is translated to `to`.
        onnx_type = node.attr["dtype"]
        shape = node.attr["shape"]
        # TODO: For now, this only permit to support NHWC data format of TF
        if cls._channel_last:
          if len(shape) >= 3:
            shape.insert(1, shape.pop())
        input_proto = make_tensor_value_info(node.name,
                                             onnx_type,
                                             shape)
        inputs_proto.append(input_proto)
      elif node.op == "Const":
        const_dim = len(node.attr["value"].shape)
        consts[node.name] = node.attr["value"]
        raw_values = ([node.attr["value"].tolist()]
                      if const_dim == 0
                      else node.attr["value"].flatten().tolist())
        if const_dim == 0:
            values = [node.attr["value"]]
        else:
            values = node.attr["value"]
        shape = list(np.array(values).shape)
        if cls._channel_last:
          if len(shape) >= 3:
            shape.insert(1, shape.pop())
        consts_proto.append(make_tensor(
                            name=node.name,
                            data_type=node.attr["dtype"],
                            dims=shape,
                            vals=raw_values))
        input_proto = make_tensor_value_info(node.name,
                                             node.attr["dtype"],
                                             shape)
        inputs_proto.append(input_proto)
      elif node.op in TF_OP_STR_TO_ONNX_OP.keys():
        # Remove tensorflow-specific attrs that are not
        # needed/allowed in ONNX.
        attr_to_remove = ["_output_shapes", "T", "seed2", "Tidx"]
        node.attr = dict(filter(lambda pair: pair[0]
                                not in attr_to_remove, node.attr.items()))

        node_output = node.name
        ops_proto.append(make_node(TF_OP_STR_TO_ONNX_OP[node.op],
                                   node.inputs,
                                   [node_output],
                                   name=node.name,
                                   **node.attr))
      else:
        handler_name = "handle_" + op_name_to_lower(node.op)

        # Check if specialized handler exists.
        if handler_name in dir(cls):
          input_node = get_nodes_by_name(graph_def.node, node.inputs)
          method_to_call = getattr(cls, handler_name)
          ops_proto.append(method_to_call(node, consts, input_node))
        else:
          raise NotImplementedError("{} op is not implemented, handler {} do not exists.".format(node.op, handler_name))

    output = TensorflowNode(output)
    # making output proto
    # TODO: deal with multi-output case.
    # TODO: default to BOOL, cf.
    # https://github.com/tensorflow/tensorflow/issues/14769
    output_onnx_type = output.attr.get("T", TensorProto.BOOL)
    out_shape = output.attr["_output_shapes"][0]
    if cls._channel_last:
      if len(out_shape) >= 3:
        out_shape.insert(1, out_shape.pop())
    output_proto = make_tensor_value_info(output.name,
                                          output_onnx_type,
                                          out_shape)
    return make_graph(ops_proto,
                      name,
                      inputs_proto,
                      [output_proto],
                      consts_proto)

  @classmethod
  def _bin_op(cls, node, onnx_op):
    node.attr["broadcast"] = 1
    return helper.make_node(
            onnx_op, node.inputs, [node.name], name=node.name, broadcast=1)

  @classmethod
  def handle_logical_and(cls, node, consts, input_node):
    return cls._bin_op(node, "And")

  @classmethod
  def handle_logical_or(cls, node, consts, input_node):
    return cls._bin_op(node, "Or")

  @classmethod
  def handle_pad(cls, node, consts, input_node):
    assert node.inputs[1] in consts.keys()
    supported_modes = ["constant", "reflect"]
    mode = node.attr.get("mode", "constant")
    assert mode.lower() in supported_modes
    if mode in ["constant"]:
      value = node.attr.get("constant_values", 0.0)
    else:
      value = 0.0
    # Pads from (n, 2) (tf) to (2, n) (onnx)
    pads = np.transpose(consts[node.inputs[1]])
    # Arrange pads to the according data format 
    if cls._channel_last:
      rank = pads.shape[1] # n
      if rank >= 3:
        old_pads = pads[:][:]
        for i, l in enumerate(old_pads[0]):
          for j, p in enumerate(l):
            if mode in ["reflect"]:
              assert p < input_node[0].attr["_output_shapes"][0][j]
            new_j = j % (rank-1) + 1 if j != 0 else 0
            pads[i][new_j] = p

    pads = pads.flatten()
    return helper.make_node(
            "Pad",
            [node.inputs[0]],
            [node.name],
            name=node.name,
            pads=pads,
            mode=mode,
            value=value)

  @classmethod
  def handle_random_standard_normal(cls, node, consts, input_node):
    """ Tensorflow does not have a generic random_normal op.
        The generic random_normal op is translated into a scaled
        and offsetted random standard normal op.
    """
    shape = node.attr["_output_shapes"][0]
    # Arrange shape to the according data format 
    if cls._channel_last:
      rank = len(shape)
      if rank >= 3:
        old_shape = shape[:]
        for i, s in enumerate(old_shape):
          j = i % (rank-1) + 1 if i != 0 else 0
          shape[j] = s
    return helper.make_node(
            "RandomNormal",
            [],
            [node.name],
            dtype=node.attr["dtype"],
            seed=node.attr["seed"],
            mean=0.0,
            scale=1.0,
            shape=shape)

  @classmethod
  def handle_random_uniform(cls, node, consts, input_node):
    """ Tensorflow does not have a generic random_uniform op.
        The generic random_uniform op is translated into a scaled
        and offsetted random standard uniform op.
    """
    shape = node.attr["_output_shapes"][0]
    # Arrange shape to the according data format 
    if cls._channel_last:
      rank = len(shape)
      if rank >= 3:
        old_shape = shape[:]
        for i, s in enumerate(shape):
          j = i % (rank-1) + 1 if i != 0 else 0
          shape[j] = s
    return helper.make_node(
            "RandomUniform",
            [],
            [node.name],
            dtype=node.attr["dtype"],
            seed=node.attr["seed"],
            high=1.0,
            low=0.0,
            shape=shape)

  @classmethod
  def _reduce_op(cls, op, node, consts, input_node):
    assert node.inputs[1] in consts.keys()
    axes = consts[node.inputs[1]]
    # Arrange axes to the according data format 
    if cls._channel_last:
      rank = len(input_node[0].attr["_output_shapes"][0])
      if rank >= 3:
        old_axes = axes[:]
        for i, a in enumerate(old_axes):
          assert a >= -rank and a < rank
          axes[i] = a % (rank-1) + 1 if a != 0 else 0
    return helper.make_node(op,
                            [node.inputs[0]],
                            [node.name],
                            axes=axes,
                            keepdims=node.attr.get("keep_dims", 1))

  @classmethod
  def handle_max(cls, node, consts, input_node):
    return cls._reduce_op("ReduceMax", node, consts, input_node)

  @classmethod
  def handle_mean(cls, node, consts, input_node):
    return cls._reduce_op("ReduceMean", node, consts, input_node)

  @classmethod
  def handle_min(cls, node, consts, input_node):
    return cls._reduce_op("ReduceMin", node, consts, input_node)

  @classmethod
  def handle_prod(cls, node, consts, input_node):
    return cls._reduce_op("ReduceProd", node, consts, input_node)

  @classmethod
  def handle_sum(cls, node, consts, input_node):
    return cls._reduce_op("ReduceSum", node, consts, input_node)

  @classmethod
  def handle_reshape(cls, node, consts, input_node):
    assert node.inputs[1] in consts.keys()
    shape = consts[node.inputs[1]]
    # Arrange shape to the according data format 
    if cls._channel_last:
      input_shape = input_node[0].attr["_output_shapes"][0]
      rank = len(input_shape)
      if rank >= 3:
        if (input_shape[0] == shape[0]) and (input_shape[-1] == shape[-1]):
          # In this case, the N and C of the data format is keeped
          old_shape = shape[:]
          for i, s in enumerate(old_shape):
            j = i % (rank-1) + 1 if i != 0 else 0
            shape[j] = s
        else:
          raise NotImplementedError("Reshaping other dimension than spacial dimensions is not implemented.")
    return helper.make_node("Reshape",
                            [node.inputs[0]],
                            [node.name],
                            shape=shape)

  @classmethod
  def handle_split_v(cls, node, consts, input_node):
    assert node.inputs[1] in consts.keys()
    split = consts[node.inputs[1]]
    # Arrange split to the according data format 
    if cls._channel_last:
      rank = len(split)
      if rank >= 3:
        old_split = split[:]
        for i, p in enumerate(old_split):
          assert i >= -rank and i < rank
          j = i % (rank-1) + 1 if i != 0 else 0
          split[j] = p
    axis = int(consts[node.inputs[2]])
    # Arrange axis to the according data format 
    if cls._channel_last:
      rank = len(input_node[0].attr["_output_shapes"][0])
      assert axis >= -rank and axis < rank
      if rank >= 3:
        axis = axis % (rank-1) + 1 if axis != 0 else 0
    output_names = [node.name + ":{}".format(i) if i>0 else node.name for i in range(len(split))]
    return helper.make_node("Split",
                            [node.inputs[0]],
                            output_names,
                            split=split,
                            axis=axis)

  @classmethod
  def handle_squeeze(cls, node, consts, input_node):
    assert "squeeze_dims" in node.attr.keys(), ("Squeeze dims have to be"
      "specified")
    axes = node.attr["squeeze_dims"]
    # Arrange data to the according data format 
    if cls._channel_last:
      rank = len(input_node[0].attr["_output_shapes"][0])
      if rank >= 3:
        old_axes = axes[:]
        for i, a in enumerate(old_axes):
          assert a >= -rank and a < rank
          axes[i] = a % (rank-1) + 1 if a != 0 else 0
    return helper.make_node("Squeeze",
                            [node.inputs[0]],
                            [node.name],
                            axes=axes)
  @classmethod
  def handle_sub(cls, node, consts, input_node):
    return cls._bin_op(node, "Sub")

  @classmethod
  def handle_transpose(cls, node, consts, input_node):
    perm = consts[node.inputs[1]]
    # Arrange perm to the according data format 
    if cls._channel_last:
      rank = len(perm)
      if rank >= 3:
        old_perm = perm[:]
        for i, p in enumerate(old_perm):
          assert p >= -rank and p < rank
          new_i = i % (rank-1) + 1 if i != 0 else 0
          new_p = p % (rank-1) + 1 if p != 0 else 0
          perm[new_i] = new_p
    return helper.make_node("Transpose",
                            [node.inputs[0]],
                            [node.name],
                            perm=perm)

  @classmethod
  def handle_logical_xor(cls, node, consts, input_node):
    return cls._bin_op(node, "Xor")

  @classmethod
  def handle_concat_v2(cls, node, consts, input_node):
    assert node.inputs[-1] in consts.keys()
    axis = int(consts[node.inputs[-1]])
    # Arrange axis to the according data format 
    if cls._channel_last:
      rank = len(input_node[0].attr["_output_shapes"][0])
      assert axis >= -rank and axis < rank
      if rank >= 3:
        axis = axis % (rank-1) + 1 if axis != 0 else 0
    return helper.make_node("Concat",
                            inputs=node.inputs[0:-1],
                            outputs=[node.name],
                            axis=axis)

convert_graph = TensorflowFrontend.tensorflow_graph_to_onnx_graph
