"""Frontend for exporting Tensorflow graph to ONNX graph

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import importlib
from itertools import chain

import tensorflow as tf
import numpy as np
from onnx_tf.common import (
  TF_TYPE_TO_ONNX_TYPE,
  TF_OP_STR_TO_ONNX_OP,
  TF_ATTR_TO_ONNX_ATTR,
  TF_ATTR_TO_REMOVE,
  get_tf_shape_as_list,
  op_name_to_lower,
)
from onnx_tf.opset_version import frontend_opset_version
from onnx import defs, helper
from onnx.helper import (
  make_tensor_value_info,
  make_tensor,
  make_graph,
  make_node,
)
from onnx.onnx_pb2 import GraphProto, TensorProto, AttributeProto
from tensorflow.python.framework.tensor_util import MakeNdarray

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

class TensorflowFrontendBase(object):
  """ Tensorflow Frontend for ONNX
  """

  @classmethod
  def tensorflow_graph_to_onnx_graph(cls, graph_def, output, opset=0, name="graph"):
    """Function that converts a tensorflow graph to an onnx graph.

    Args:
        graph_def: Tensorflow Graph Proto object.
        output: A Tensorflow NodeDef object specifying which node
          to be taken as output of the ONNX graph.
        opset: Opset version of the operator set. Default 0 means using latest version.
        name: The name of the output ONNX Graph.

    Returns:
        The equivalent ONNX Graph Proto object.

    """

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
        shape = np.array(values).shape
        consts_proto.append(make_tensor(
                            name=node.name,
                            data_type=node.attr["dtype"],
                            dims=shape,
                            vals=raw_values))
        input_proto = make_tensor_value_info(node.name,
                                             node.attr["dtype"],
                                             shape)
        inputs_proto.append(input_proto)
      else:
        handler_name = "handle_" + op_name_to_lower(node.op)

        versions = frontend_opset_version[op_name_to_lower(node.op)]

        opset = defs.onnx_opset_version() if opset == 0 else opset
        if opset == 0:
          version = max(versions)
        else:
          versions = sorted(versions + [opset])
          version = versions[max([i for i, v in enumerate(versions) if v == opset]) - 1]

        frontend = importlib.import_module('onnx_tf.frontends.frontend_v{}'.format(version)).TensorflowFrontend

        # Check if specialized handler exists.
        if handler_name in dir(frontend):
          method_to_call = getattr(frontend, handler_name)
          ops_proto.append(method_to_call(node, consts=consts))
        elif node.op in TF_OP_STR_TO_ONNX_OP.keys():
          # Remove tensorflow-specific attrs that are not
          # needed/allowed in ONNX.
          node.attr = dict(filter(lambda pair: pair[0]
                                               not in TF_ATTR_TO_REMOVE, node.attr.items()))

          node_output = node.name
          ops_proto.append(make_node(TF_OP_STR_TO_ONNX_OP[node.op],
                                     node.inputs,
                                     [node_output],
                                     name=node.name,
                                     **node.attr))
        else:
          raise NotImplementedError("{} op is not implemented.".format(node.op))

    output = TensorflowNode(output)
    # making output proto
    # TODO: deal with multi-output case.
    # TODO: default to BOOL, cf.
    # https://github.com/tensorflow/tensorflow/issues/14769
    output_onnx_type = output.attr.get("T", TensorProto.BOOL)
    output_proto = []
    for i in range(len(output.attr["_output_shapes"])):
      output_name = output.name + ":{}".format(i) if i > 0 else output.name
      output_proto.append(make_tensor_value_info(output_name,
                                                 output_onnx_type,
                                                 output.attr["_output_shapes"][i]))

    inputs = list(chain.from_iterable(map(lambda p: list(p.input), ops_proto)))

    # Remove proto in inputs_proto and consts_proto if proto is not used as input in ONNX
    inputs_proto = list(filter(lambda x: x.name in inputs, inputs_proto))
    consts_proto = list(filter(lambda x: x.name in inputs, consts_proto))

    return make_graph(ops_proto,
                      name,
                      inputs_proto,
                      output_proto,
                      consts_proto)

  @classmethod
  def _bin_op(cls, node, onnx_op):
    node.attr["broadcast"] = 1
    return helper.make_node(
            onnx_op, node.inputs, [node.name], name=node.name, broadcast=1)

  @classmethod
  def _reduce_op(cls, op, node, **kwargs):
    consts = kwargs["consts"]
    assert node.inputs[1] in consts.keys()
    axes = consts[node.inputs[1]]
    return helper.make_node(op,
                            [node.inputs[0]],
                            [node.name],
                            axes=axes,
                            keepdims=node.attr.get("keep_dims", 1))

convert_graph = TensorflowFrontendBase.tensorflow_graph_to_onnx_graph
