"""Frontend for exporting Tensorflow graph to ONNX graph

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import tensorflow as tf

from onnx_tf.common import (
  TF_TYPE_TO_ONNX_TYPE,
  get_tf_shape_as_list,
  op_name_to_lower,
)
from onnx import onnx_pb2, helper
from onnx.onnx_pb2 import GraphProto, TensorProto, AttributeProto

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
      if node.op == "Placeholder":
        # Tensorflow requires dtype to be known.
        onnx_type = TF_TYPE_TO_ONNX_TYPE[tf.as_dtype(node.attr["dtype"].type)]
        shape = get_tf_shape_as_list(node.attr["shape"].shape.dim)
        input_proto = helper.make_tensor_value_info(node.name,
                                                    onnx_type,
                                                    shape)
        inputs_proto.append(input_proto)
      else:
        handler_name = "handle_" + op_name_to_lower(node.op)

        # Check if specialized handler exists.
        if handler_name in dir(cls):
          method_to_call = getattr(cls, handler_name)
          ops_proto.append(method_to_call(node))

    # making output proto
    output_onnx_type = TF_TYPE_TO_ONNX_TYPE[output.attr["T"].type]
    output_shape = get_tf_shape_as_list(output.attr["_output_shapes"].list.shape[0].dim)
    output_proto = helper.make_tensor_value_info(output.name,
                                                 output_onnx_type,
                                                 output_shape)

    return helper.make_graph(ops_proto,
                            name,
                            inputs_proto,
                            [output_proto])

  @classmethod
  def handle_relu(cls, node_proto):
    return helper.make_node(
            "Relu", [str(node_proto.input[0])], [node_proto.name], name=node_proto.name)

convert_graph = TensorflowFrontend.tensorflow_graph_to_onnx_graph