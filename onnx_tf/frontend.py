"""Frontend for exporting Tensorflow graph to ONNX graph

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import importlib
import inspect
from itertools import chain
import sys
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.tensor_util import MakeNdarray
from tensorflow.core.framework.attr_value_pb2 import AttrValue

# Define long type for Python 3:
if sys.version_info > (3,):
  long = int

from onnx_tf.common import (
    TF_ATTR_TO_ONNX_ATTR,
    get_attribute_value,
    get_tf_shape_as_list,
)
from onnx import defs
from onnx import TensorProto
from onnx import ValueInfoProto
from onnx.helper import (
    make_tensor_value_info,
    make_tensor,
    make_graph,
    make_model,
    make_opsetid,
)
from onnx.helper import mapping

from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.common.handler_helper import get_all_frontend_handlers
from onnx_tf.common import exception
from onnx_tf.common import data_type


class TensorflowNode(object):
  # Keyed by old attribute names.
  attr_translator = {
    "_output_shapes": lambda self, x: list(map(lambda shape: get_tf_shape_as_list(shape.dim), x.list.shape)),
    "shape": lambda self, x: get_tf_shape_as_list(x.shape.dim),
    "T": lambda self, x: data_type.tf2onnx(x.type),
    "dtype": lambda self, x: data_type.tf2onnx(x.type),
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

      if isinstance(self.attr[new_key], AttrValue):
        self.attr[new_key] = get_attribute_value(self.attr[new_key])


class TensorflowFrontendBase(object):
  """ Tensorflow Frontend for ONNX
  """

  @classmethod
  def tensorflow_graph_to_onnx_graph(cls,
                                     graph_def,
                                     output,
                                     opset=(("", 0),),
                                     name="graph",
                                     ignore_unimplemented=False):
    """Converts a Tensorflow Graph Proto to an ONNX graph

    This function converts a Tensorflow Graph proto to an equivalent
    representation of ONNX graph.

    :param graph_def: Tensorflow Graph Proto object.
    :param output: A Tensorflow NodeDef object specifying which node
      to be taken as output of the ONNX graph.
    :param opset: Opset, which should be ((str domain: int version number),).
    :param name: The name of the output ONNX Graph.
    :param ignore_unimplemented: Convert to ONNX model and ignore all the operators
      that are not currently supported by onnx-tensorflow.
      This is an experimental feature. By enabling this feature,
      the graph would not be guaranteed to match the ONNX specifications.

    :returns: The equivalent ONNX Graph Proto object.
    """

    handlers = get_all_frontend_handlers()

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

    # This list holds the protobuf objects of type ValueInfoProto
    # representing the all nodes' outputs to the converted ONNX graph.
    value_info_proto = []

    # A map holds nodes name and new data type. Will be used to
    # process protos to match ONNX type constraints.
    data_type_cast_map = {}

    node_tup = [(node.name, TensorflowNode(node)) for node in graph_def.node]

    for name, node in node_tup:

      if node.op == "Placeholder":
        # Tensorflow requires dtype to be known.
        # TODO: currently `dtype` is translated to `to`.
        onnx_type = node.attr["dtype"]
        shape = node.attr["shape"]
        input_proto = make_tensor_value_info(name, onnx_type, shape)
        inputs_proto.append(input_proto)
      elif node.op == "Const":
        const_dim = len(node.attr["value"].shape)

        consts[name] = node.attr["value"]
        raw_values = ([node.attr["value"].tolist()] if const_dim == 0 else
                      node.attr["value"].flatten().tolist())
        if const_dim == 0:
          values = [node.attr["value"]]
        else:
          values = node.attr["value"]
        shape = np.array(values).shape
        consts_proto.append(
            make_tensor(
                name=name,
                data_type=node.attr["dtype"],
                dims=shape,
                vals=raw_values))
        input_proto = make_tensor_value_info(name, node.attr["dtype"], shape)
        inputs_proto.append(input_proto)
      else:
        for i in range(len(node.attr["_output_shapes"])):
          node_name = node.name + ":{}".format(i) if i > 0 else node.name
          value_info_proto.append(
              make_tensor_value_info(node_name,
                                     node.attr.get("T", TensorProto.BOOL),
                                     node.attr["_output_shapes"][i]))

        splitted_op_name = node.op.split(".")
        op_domain = "" if len(splitted_op_name) == 1 else ".".join(
            splitted_op_name[:-1])
        op_name = splitted_op_name[-1]

        opset_dict = {}
        for domain, version in opset:
          if domain == "ai.onnx":
            domain = ""
          opset_dict[domain] = version
          assert isinstance(version, (int, long)) and (
              version <= defs.C.schema_version_map()[domain][1]) and (
                  version >= defs.C.schema_version_map()[domain][0]
              ), "Opset should be an int in ({}, {}), but {}: {}".format(
                  defs.C.schema_version_map()[domain][0],
                  defs.C.schema_version_map()[domain][1], type(version),
                  version)

        opset_ver = opset_dict[op_domain]
        handler = handlers.get(op_name, None)

        if handler:
          node = handler.handle(
              node,
              opset_ver,
              consts=consts,
              node_dict=dict(node_tup),
              data_type_cast_map=data_type_cast_map)
          if isinstance(node, list):
            ops_proto.extend(node)
          else:
            ops_proto.append(node)
        else:
          exception.OP_NOT_IMPL_EXCEPT(node.op)
          ops_proto.append(FrontendHandler.make_node(node, should_check=False))

    output = TensorflowNode(output)
    # making output proto
    # TODO: deal with multi-output case.
    # TODO: default to BOOL, cf.
    # https://github.com/tensorflow/tensorflow/issues/14769
    output_onnx_type = output.attr.get("T", TensorProto.BOOL)
    output_proto = []
    for i in range(len(output.attr["_output_shapes"])):
      output_name = output.name + ":{}".format(i) if i > 0 else output.name
      output_proto.append(
          make_tensor_value_info(output_name, output_onnx_type,
                                 output.attr["_output_shapes"][i]))

    inputs = list(chain.from_iterable(map(lambda p: p.input, ops_proto)))
    outputs = list(map(lambda p: p.name, output_proto))
    in_out = inputs + outputs

    # Remove proto in inputs_proto and consts_proto
    # if proto is not used as input or an output in ONNX
    inputs_proto = list(filter(lambda x: x.name in in_out, inputs_proto))
    consts_proto = list(filter(lambda x: x.name in in_out, consts_proto))

    inputs_proto = cls._data_type_caster(inputs_proto, data_type_cast_map)
    consts_proto = cls._data_type_caster(consts_proto, data_type_cast_map)

    # TODO: currently no onnx release support value_info, thus ensuring
    # backward compatibility via try catch routine. Switch to excplicit
    # onnx version checking when value_info is supported in upcoming
    # onnx release.
    try:
      return make_graph(
          ops_proto,
          name,
          inputs_proto,
          output_proto,
          initializer=consts_proto,
          value_info=value_info_proto)
    except TypeError:
      return make_graph(
          ops_proto, name, inputs_proto, output_proto, initializer=consts_proto)

  @classmethod
  def _data_type_caster(cls, protos, data_type_cast_map):
    """Cast to a new data type if node name is in data_type_cast_map.
    Be used to process protos to match ONNX type constraints.

    :param protos: Target protos.
      TensorProto for inputs and ValueInfoProto for consts.
    :param data_type_cast_map: A {node.name: new_data_type} dict.
    :return: Processed protos.
    """
    if not data_type_cast_map:
      return protos
    result = []
    for proto in protos:
      new_proto = proto
      if proto.name in data_type_cast_map:
        new_data_type = data_type_cast_map[proto.name]
        if type(proto) == TensorProto and proto.data_type != new_data_type:
          field = mapping.STORAGE_TENSOR_TYPE_TO_FIELD[
              mapping.TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE[proto.data_type]]
          vals = getattr(proto, field)
          new_proto = make_tensor(
              name=proto.name,
              data_type=new_data_type,
              dims=proto.dims,
              vals=vals)
        elif type(
            proto
        ) == ValueInfoProto and proto.type.tensor_type.elem_type != new_data_type:
          new_proto.type.tensor_type.elem_type = new_data_type
      result.append(new_proto)
    return result

  @classmethod
  def tensorflow_graph_to_onnx_model(cls,
                                     graph_def,
                                     output,
                                     opset=0,
                                     producer_name="onnx-tensorflow",
                                     graph_name="graph",
                                     ignore_unimplemented=False):
    """Converts a Tensorflow Graph Proto to an ONNX model

    This function converts a Tensorflow Graph proto to an equivalent
    representation of ONNX model.

    :param graph_def: Tensorflow Graph Proto object.
    :param output: A string specifying the name of the output
      graph node.
    :param opset: Opset version number, list or tuple.
      Default is 0 means using latest version with domain ''.
      List or tuple items should be (str domain, int version number).
    :param producer_name: The name of the producer.
    :param graph_name: The name of the output ONNX Graph.
    :param ignore_unimplemented: Convert to ONNX model and ignore all the operators
      that are not currently supported by onnx-tensorflow.
      This is an experimental feature. By enabling this feature,
      the model would not be guaranteed to match the ONNX specifications.

    :returns: The equivalent ONNX Model Proto object.
    """

    def get_node_by_name(nodes, name):
      for node in nodes:
        if node.name == name:
          return node
      raise ValueError(
          "Node {} is not found in the graph provided".format(name))

    assert isinstance(
        opset,
        (int, long, list,
         tuple)), "opset is expected to int, list or tuple, but {}.".format(
             type(opset))
    if isinstance(opset, (int, long)):
      if opset == 0:
        opset = defs.onnx_opset_version()
      opset = [("", opset)]
    opset_imports = [make_opsetid(item[0], item[1]) for item in opset]

    output_node = get_node_by_name(graph_def.node, output)

    if "_output_shapes" not in output_node.attr:
      # Add infer_shapes to GraphDef
      with tf.Graph().as_default():
        with tf.Session(
            config=tf.ConfigProto(
                graph_options=tf.GraphOptions(infer_shapes=True))) as sess:
          tf.import_graph_def(graph_def, name="")
        graph_def = sess.graph_def
      output_node = get_node_by_name(graph_def.node, output)

    exception.USE_WARNING = ignore_unimplemented
    onnx_graph = cls.tensorflow_graph_to_onnx_graph(
        graph_def, output_node, opset, graph_name, ignore_unimplemented)
    onnx_model = make_model(
        onnx_graph, producer_name=producer_name, opset_imports=opset_imports)

    return onnx_model


convert_graph = TensorflowFrontendBase.tensorflow_graph_to_onnx_graph

tensorflow_graph_to_onnx_model = TensorflowFrontendBase.tensorflow_graph_to_onnx_model
