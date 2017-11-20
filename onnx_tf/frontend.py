"""Frontend for exporting Tensorflow graph to ONNX graph

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re

from onnx import onnx_pb2, helper
from onnx.onnx_pb2 import GraphProto, TensorProto, AttributeProto

class TensorflowFrontend(object):
  """ Tensorflow Frontend for ONNX
  """

  # refer to
  # https://github.com/tensorflow/tensorflow/blob/f284c708a8f3e2672d41dd3e7f6f03b9d26f0c80/tensorflow/core/framework/types.proto

  onnx_types = [
  "invalid",
  TensorProto.FLOAT,
  TensorProto.DOUBLE,
  # DT_INT32 = 3;
  # DT_UINT8 = 4;
  # DT_INT16 = 5;
  # DT_INT8 = 6;
  # DT_STRING = 7;
  # DT_COMPLEX64 = 8;  // Single-precision complex
  # DT_INT64 = 9;
  # DT_BOOL = 10;
  # DT_QINT8 = 11;     // Quantized int8
  # DT_QUINT8 = 12;    // Quantized uint8
  # DT_QINT32 = 13;    // Quantized int32
  # DT_BFLOAT16 = 14;  // Float32 truncated to 16 bits.  Only for cast ops.
  # DT_QINT16 = 15;    // Quantized int16
  # DT_QUINT16 = 16;   // Quantized uint16
  # DT_UINT16 = 17;
  # DT_COMPLEX128 = 18;  // Double-precision complex
  # DT_HALF = 19;
  # DT_RESOURCE = 20;
  # DT_VARIANT = 21;  // Arbitrary C++ data types
  # DT_UINT32 = 22;
  # DT_UINT64 = 23;
  ]

  @classmethod
  def op_name_to_lower(cls, name):
    return re.sub('(?<!^)(?=[A-Z])', '_', name).lower()

  @classmethod
  def tensorflow_graph_to_onnx_graph(cls, graph_def, output):
    inputs_proto = []
    ops_proto = []

    for node in graph_def.node:
      if node.op == "Placeholder":
        shape = map(lambda x: x.size, list(node.attr["shape"].shape.dim))
        inputs_proto.append(helper.make_tensor_value_info(node.name,
                                                          cls.onnx_types[node.attr["dtype"].type],
                                                          shape))
      else:
        handler_name = "handle_" + cls.op_name_to_lower(node.op)

        # Check if specialized handler exists.
        if handler_name in dir(cls):
          method_to_call = getattr(cls, handler_name)
          ops_proto.append(method_to_call(node))

    return helper.make_graph(ops_proto,
                            "onnx_graph",
                            inputs_proto,
                            [helper.make_tensor_value_info(output.name,
                                                          cls.onnx_types[output.attr["T"].type],
                                                          [10])])

  @classmethod
  def handle_relu(cls, node_proto):
    return helper.make_node(
            "Relu", [str(node_proto.input[0])], [node_proto.name], name=node_proto.name)

convert_graph = TensorflowFrontend.tensorflow_graph_to_onnx_graph