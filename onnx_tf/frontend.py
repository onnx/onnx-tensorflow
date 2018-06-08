"""Frontend for exporting Tensorflow graph to ONNX graph

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
from itertools import chain

import numpy as np
from onnx import defs
from onnx import TensorProto
from onnx import ValueInfoProto
from onnx.helper import make_graph
from onnx.helper import make_model
from onnx.helper import make_opsetid
from onnx.helper import make_tensor
from onnx.helper import make_tensor_value_info
from onnx.helper import mapping
from onnx.optimizer import optimize
import tensorflow as tf
from tensorflow.core.framework.attr_value_pb2 import AttrValue

from onnx_tf.common import attr_converter
from onnx_tf.common import attr_translator
from onnx_tf.common import exception
from onnx_tf.common.handler_helper import get_all_frontend_handlers
from onnx_tf.common import IS_PYTHON3
from onnx_tf.handlers.frontend_handler import FrontendHandler

# Define long type for Python 3:
if IS_PYTHON3:
  long = int


class TensorflowNode(object):

  def __init__(self, node_proto):
    # storing a reference to the original protobuf object
    self.node_proto = node_proto
    self.name = node_proto.name
    self.inputs = list(node_proto.input)
    self.attr = {}

    for key, val in node_proto.attr.items():
      new_val = attr_translator.translate_tf(key, val)

      if isinstance(new_val, AttrValue):
        new_val = attr_converter.convert_tf(new_val)

      self.attr[key] = new_val

    splitted_op_name = node_proto.op.split(".")
    self.domain = "" if len(splitted_op_name) == 1 else ".".join(
        splitted_op_name[:-1])
    self.op_type = splitted_op_name[-1]


class OnnxGraph(object):
  """ A helper class for making ONNX graph.
  This class holds all information ONNX graph needs.
  """

  def __init__(self, name):
    self._name = name
    self._inputs_proto = []
    self._outputs_proto = []
    self._nodes_proto = []
    self._consts = {}
    self._consts_proto = []
    self._value_info_proto = []
    self._data_type_cast_map = {}

  # This list holds the protobuf objects of type ValueInfoProto
  # representing the input to the converted ONNX graph.
  @property
  def inputs_proto(self):
    return self._inputs_proto

  @inputs_proto.setter
  def inputs_proto(self, inputs_proto):
    self._inputs_proto = inputs_proto

  @property
  def all_node_inputs(self):
    return list(chain.from_iterable(map(lambda p: p.input, self._nodes_proto)))

  @property
  def outputs(self):
    return list(map(lambda p: p.name, self._outputs_proto))

  @property
  def outputs_proto(self):
    return self._outputs_proto

  # This list holds the protobuf objects of type NodeProto
  # representing the ops in the converted ONNX graph.
  @property
  def nodes_proto(self):
    return self._nodes_proto

  @nodes_proto.setter
  def nodes_proto(self, nodes_proto):
    self._nodes_proto = nodes_proto

  # This dictionary contains a map from the name of the constant
  # op to the array of values it holds. This is useful because
  # tensorflow is less eager to know about input values at
  # graph construction time than ONNX. That is to say, some ONNX
  # attributes are input tensors in TF. This dictionary extracts
  # those values of constant tensors that are known at graph
  # construction time.
  @property
  def consts(self):
    return self._consts

  @consts.setter
  def consts(self, consts):
    self._consts = consts

  # Sometimes the constants are used as inputs to ops. This list
  # holds initializers that creates global constant tensors available
  # to be accessed by ops as inputs (as oppose to attributes which
  # is supplied by the `consts` map above).
  @property
  def consts_proto(self):
    return self._consts_proto

  @consts_proto.setter
  def consts_proto(self, consts_proto):
    self._consts_proto = consts_proto

  # A map holds nodes name and new data type. Will be used to
  # process protos to match ONNX type constraints.
  @property
  def data_type_cast_map(self):
    return self._data_type_cast_map

  @data_type_cast_map.setter
  def data_type_cast_map(self, data_type_cast_map):
    self._data_type_cast_map = data_type_cast_map

  # This list holds the protobuf objects of type ValueInfoProto
  # representing the all nodes' outputs to the converted ONNX graph.
  @property
  def value_info_proto(self):
    return self._value_info_proto

  def add_input_proto(self, node):
    onnx_type = node.attr["dtype"]
    shape = node.attr["shape"] if node.op_type != "Const" else node.attr[
        'value'].shape
    input_proto = make_tensor_value_info(node.name, onnx_type, shape)
    self._inputs_proto.append(input_proto)

  def add_output_proto(self, node):
    output_onnx_type = node.attr.get("T", TensorProto.BOOL)
    for i, output_shape in enumerate(node.attr["_output_shapes"]):
      output_name = node.name + ":{}".format(i) if i > 0 else node.name
      self._outputs_proto.append(
          make_tensor_value_info(output_name, output_onnx_type, output_shape))

  def add_node_proto(self, node_proto):
    if not isinstance(node_proto, (list, tuple)):
      node_proto = [node_proto]
    self._nodes_proto.extend(node_proto)

  def add_const(self, node):
    self._consts[node.name] = node.attr["value"]

  def add_const_proto(self, node):
    const_dim = len(node.attr["value"].shape)

    if const_dim == 0:
      raw_values = [node.attr["value"].tolist()]
      values = [node.attr["value"]]
    else:
      raw_values = node.attr["value"].flatten().tolist()
      values = node.attr["value"]

    shape = np.array(values).shape
    const_proto = make_tensor(
        name=node.name,
        data_type=node.attr["dtype"],
        dims=shape,
        vals=raw_values)
    self._consts_proto.append(const_proto)

  def add_value_info_proto(self, node):
    node_onnx_type = node.attr.get("T", TensorProto.BOOL)
    for i, output_shape in enumerate(node.attr["_output_shapes"]):
      node_name = node.name + ":{}".format(i) if i > 0 else node.name
      value_info_proto = make_tensor_value_info(node_name, node_onnx_type,
                                                output_shape)
      self._value_info_proto.append(value_info_proto)

  # Remove proto in inputs_proto and consts_proto
  # if proto is not used as input or an output in ONNX
  def _clean_graph(self):
    in_out = self.all_node_inputs + self.outputs
    self._inputs_proto = list(
        filter(lambda x: x.name in in_out, self.inputs_proto))
    self._consts_proto = list(
        filter(lambda x: x.name in in_out, self.consts_proto))

  def _fix_data_type(self):
    self.inputs_proto = self._data_type_caster(self.inputs_proto,
                                               self.data_type_cast_map)
    self.consts_proto = self._data_type_caster(self.consts_proto,
                                               self.data_type_cast_map)

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

  def make_graph_proto(self):
    self._clean_graph()
    self._fix_data_type()

    if IS_PYTHON3:
      params = list(inspect.signature(make_graph).parameters.keys())
    else:
      params = inspect.getargspec(make_graph).args

    kwargs = {
        "initializer": self.consts_proto,
        "value_info": self.value_info_proto
    }

    return make_graph(self.nodes_proto, self._name, self.inputs_proto,
                      self.outputs_proto,
                      **dict([(k, kwargs[k]) for k in kwargs if k in params]))


class TensorflowFrontend(object):
  """ Tensorflow Frontend for ONNX
  """

  @classmethod
  def tensorflow_graph_to_onnx_graph(cls,
                                     graph_def,
                                     output,
                                     opset=((defs.ONNX_DOMAIN,
                                             defs.onnx_opset_version()),),
                                     name="graph",
                                     ignore_unimplemented=False):
    """Converts a Tensorflow Graph Proto to an ONNX graph

    This function converts a Tensorflow Graph proto to an equivalent
    representation of ONNX graph.

    :param graph_def: Tensorflow Graph Proto object.
    :param output: List of Tensorflow NodeDef object specifying which nodes
      to be taken as outputs of the ONNX graph.
    :param opset: Opset, which should be ((str domain: int version number),).
    :param name: The name of the output ONNX Graph.
    :param ignore_unimplemented: Convert to ONNX model and ignore all the operators
      that are not currently supported by onnx-tensorflow.
      This is an experimental feature. By enabling this feature,
      the graph would not be guaranteed to match the ONNX specifications.

    :returns: The equivalent ONNX Graph Proto object.
    """
    onnx_graph = OnnxGraph(name)
    exception.IGNORE_UNIMPLEMENTED = ignore_unimplemented

    opset_dict = {}
    for domain, version in opset:
      if domain == "ai.onnx":
        domain = defs.ONNX_DOMAIN
      opset_dict[domain] = version

    handlers = get_all_frontend_handlers(opset_dict)

    node_tup = [(node.name, TensorflowNode(node)) for node in graph_def.node]
    for name, node in node_tup:

      if node.op_type == "Placeholder":
        onnx_graph.add_input_proto(node)
      elif node.op_type == "Const":
        onnx_graph.add_const(node)
        onnx_graph.add_const_proto(node)
        onnx_graph.add_input_proto(node)
      else:
        onnx_graph.add_value_info_proto(node)
        handler = handlers.get(node.domain, {}).get(node.op_type, None)
        node_proto = None
        if handler:
          node_proto = handler.handle(
              node,
              consts=onnx_graph.consts,
              node_dict=dict(node_tup),
              data_type_cast_map=onnx_graph.data_type_cast_map)
        else:
          exception.OP_UNIMPLEMENTED_EXCEPT(
              node.op_type,
              domain=None if node.domain in handlers else node.domain)

        if node_proto is None:
          node_proto = FrontendHandler.make_node_from_tf_node(
              node, op_type=node.op_type, should_check=False)
        onnx_graph.add_node_proto(node_proto)

    for o in output:
      output_node = TensorflowNode(o)
      onnx_graph.add_output_proto(output_node)

    return onnx_graph.make_graph_proto()

  @classmethod
  def tensorflow_graph_to_onnx_model(cls,
                                     graph_def,
                                     output,
                                     opset=0,
                                     producer_name="onnx-tensorflow",
                                     graph_name="graph",
                                     ignore_unimplemented=False,
                                     optimizer_passes=None):
    """Converts a Tensorflow Graph Proto to an ONNX model

    This function converts a Tensorflow Graph proto to an equivalent
    representation of ONNX model.

    :param graph_def: Tensorflow Graph Proto object.
    :param output: List of string or a string specifying the name
      of the output graph node.
    :param opset: Opset version number, list or tuple.
      Default is 0 means using latest version with domain ''.
      List or tuple items should be (str domain, int version number).
    :param producer_name: The name of the producer.
    :param graph_name: The name of the output ONNX Graph.
    :param ignore_unimplemented: Convert to ONNX model and ignore all the operators
      that are not currently supported by onnx-tensorflow.
      This is an experimental feature. By enabling this feature,
      the model would not be guaranteed to match the ONNX specifications.
    :param optimizer_passes: List of optimization names c.f.
      https://github.com/onnx/onnx/blob/master/onnx/optimizer.py for available
      optimization passes.

    :returns: The equivalent ONNX Model Proto object.
    """

    def get_node_by_name(nodes, name):
      for node in nodes:
        if node.name == name:
          return node
      raise ValueError(
          "Node {} is not found in the graph provided".format(name))

    if not isinstance(opset, (int, long, list, tuple)):
      raise TypeError("opset is expected to int, list or tuple, but {}.".format(
          type(opset)))
    if isinstance(opset, (int, long)):
      opset = [(defs.ONNX_DOMAIN, opset or defs.onnx_opset_version())]
    opset_imports = [make_opsetid(item[0], item[1]) for item in opset]

    if not isinstance(output, (list, tuple)):
      output = [output]

    output_nodes = [get_node_by_name(graph_def.node, o) for o in output]

    if "_output_shapes" not in output_nodes[0].attr:
      # Add infer_shapes to GraphDef
      graph_def = cls._add_infer_shapes(graph_def)
      output_nodes = [get_node_by_name(graph_def.node, o) for o in output]

    onnx_graph = cls.tensorflow_graph_to_onnx_graph(
        graph_def, output_nodes, opset, graph_name, ignore_unimplemented)
    onnx_model = make_model(
        onnx_graph, producer_name=producer_name, opset_imports=opset_imports)

    if isinstance(optimizer_passes, (list, tuple)) and optimizer_passes:
      onnx_model = optimize(onnx_model, optimizer_passes)

    return onnx_model

  @staticmethod
  def _add_infer_shapes(graph_def):
    with tf.Graph().as_default():
      with tf.Session(
          config=tf.ConfigProto(
              graph_options=tf.GraphOptions(infer_shapes=True))) as sess:
        tf.import_graph_def(graph_def, name="")
      return sess.graph_def


convert_graph = TensorflowFrontend.tensorflow_graph_to_onnx_graph

tensorflow_graph_to_onnx_model = TensorflowFrontend.tensorflow_graph_to_onnx_model
