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
  get_attribute_value,
  get_tf_shape_as_list,
  op_name_to_lower,
)
from onnx_tf.opset_version import frontend_tf_opset_version
from onnx import defs, helper
from onnx.helper import (
  make_tensor_value_info,
  make_tensor,
  make_graph,
  make_model,
  make_node,
)
from onnx.onnx_pb2 import GraphProto, TensorProto, AttributeProto
from tensorflow.python.framework.tensor_util import MakeNdarray
from tensorflow.core.framework.attr_value_pb2 import AttrValue

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

  DEFAULT_TF_ATTR_PER_OP = {
    "Add": {"broadcast": 1},
    "And": {"broadcast": 1},
    "Div": {"broadcast": 1},
    "Equal": {"broadcast": 1},
    "Greater": {"broadcast": 1},
    "Less": {"broadcast": 1},
    "Mul": {"broadcast": 1},
    "Or": {"broadcast": 1},
    "Pow": {"broadcast": 1},
    "Sub": {"broadcast": 1},
    "Xor": {"broadcast": 1},
  }

  frontend_version_cache = {}

  @classmethod
  def tensorflow_graph_to_onnx_graph(cls, graph_def, output, opset=0, name="graph"):
    """Converts a Tensorflow Graph Proto to an ONNX graph

    This function converts a Tensorflow Graph proto to an equivalent
    representation of ONNX graph.

    :param graph_def: Tensorflow Graph Proto object.
    :param output: A Tensorflow NodeDef object specifying which node
      to be taken as output of the ONNX graph.
    :param opset: Opset version of the operator set.
      Default 0 means using latest version.
    :param name: The name of the output ONNX Graph.

    :returns: The equivalent ONNX Graph Proto object.
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
        node.attr = dict(
          map(lambda item: (item[0], get_attribute_value(item[1]) if isinstance(item[1], AttrValue) else item[1]),
              node.attr.items()))

        versions = frontend_tf_opset_version[op_name_to_lower(node.op)]

        assert isinstance(opset, int) and (opset <= defs.onnx_opset_version()) and (
            opset >= 0), "Opset should be an int less than or equal to {}, but {}: {}".format(defs.onnx_opset_version(),
                                                                                              type(opset).__name__,
                                                                                              opset)

        if opset == 0:
          version = max(versions)
        else:
          versions = sorted(versions + [opset])
          version = versions[max([i for i, v in enumerate(versions) if v == opset]) - 1]

        frontend_ver = 'frontend_v{}'.format(version)
        frontend = cls.frontend_version_cache.setdefault(frontend_ver, importlib.import_module(
          'onnx_tf.frontends.' + frontend_ver).TensorflowFrontend)

        # Check if specialized handler exists.
        if hasattr(frontend, handler_name):
          method_to_call = getattr(frontend, handler_name)
          node = method_to_call(node, consts=consts)
          if isinstance(node, list):
            ops_proto.extend(node)
          else:
            ops_proto.append(node)
        elif node.op in TF_OP_STR_TO_ONNX_OP.keys():
          # Remove tensorflow-specific attrs that are not
          # needed/allowed in ONNX.
          attr = cls.DEFAULT_TF_ATTR_PER_OP.get(node.op, {})
          attr.update(dict(filter(lambda pair: pair[0] not in TF_ATTR_TO_REMOVE, node.attr.items())))
          node.attr = attr
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
  def tensorflow_graph_to_onnx_model(cls,
                                     graph_def,
                                     output,
                                     opset=0,
                                     producer_name="onnx-tensorflow",
                                     graph_name="graph"):
    """Converts a Tensorflow Graph Proto to an ONNX model

    This function converts a Tensorflow Graph proto to an equivalent
    representation of ONNX model.

    :param graph_def: Tensorflow Graph Proto object.
    :param output: A Tensorflow NodeDef object specifying which node
      to be taken as output of the ONNX graph.
    :param opset: Opset version of the operator set.
      Default 0 means using latest version.
    :param producer_name: The name of the producer.
    :param graph_name: The name of the output ONNX Graph.

    :returns: The equivalent ONNX Model Proto object.
    """
    onnx_graph = cls.tensorflow_graph_to_onnx_graph(graph_def,
                                                    output,
                                                    opset,
                                                    graph_name)
    onnx_model = make_model(onnx_graph,
                            producer_name=producer_name,
                            opset_imports=[opset])

    return onnx_model

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

tensorflow_graph_to_onnx_model = TensorflowFrontendBase.tensorflow_graph_to_onnx_model
