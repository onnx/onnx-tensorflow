"""Frontend for exporting Tensorflow graph to ONNX graph

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json

import tensorflow as tf
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
  make_graph,
  make_node,
)
from onnx.onnx_pb2 import GraphProto, TensorProto, AttributeProto
from google.protobuf.json_format import MessageToJson

class TensorflowNode(object):

  # Keyed by old attribute names.
  attr_translator = {
    "_output_shapes": lambda self, x: list(map(lambda shape: get_tf_shape_as_list(shape.dim), x.list.shape)),
    "shape": lambda self, x: get_tf_shape_as_list(x.shape.dim),
    "T": lambda self, x: self.type_converter(x),
    "dtype": lambda self, x: self.type_converter(x),
  }

  def __init__(self, node_proto):
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
  def tensorflow_graph_to_onnx_graph(cls, graph_def, output, name="graph"):
    """Function that converts a tensorflow graph to an onnx graph.

    Args:
        graph_def: Tensorflow Graph Proto object.
        output: A Tensorflow NodeDef object specifying which node
          to be taken as output of the ONNX graph.
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

    for node in graph_def.node:
      node = TensorflowNode(node)
      if node.op == "Placeholder":
        # Tensorflow requires dtype to be known.
        # TODO: currently `dtype` is translated to `to`.
        onnx_type = node.attr["to"]
        shape = node.attr["shape"]
        input_proto = make_tensor_value_info(node.name,
                                             onnx_type,
                                             shape)
        inputs_proto.append(input_proto)

      elif node.op in TF_OP_STR_TO_ONNX_OP.keys():
        # Remove tensorflow-specific attrs that are not
        # needed/allowed in ONNX.
        attr_to_remove = ["_output_shapes", "T"]
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
          method_to_call = getattr(cls, handler_name)
          ops_proto.append(method_to_call(node))

    output = TensorflowNode(output)
    # making output proto
    # TODO: deal with multi-output case.
    output_onnx_type = output.attr["T"]
    output_proto = make_tensor_value_info(output.name,
                                          output_onnx_type,
                                          output.attr["_output_shapes"][0])

    return make_graph(ops_proto,
                      name,
                      inputs_proto,
                      [output_proto])

  # This is kept as an example, it's never used.
  @classmethod
  def handle_relu(cls, node):
    return helper.make_node(
            "Relu", node.inputs, [node.name], name=node.name)

convert_graph = TensorflowFrontend.tensorflow_graph_to_onnx_graph