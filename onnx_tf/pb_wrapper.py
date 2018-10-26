import inspect
from itertools import chain

import numpy as np
from onnx import NodeProto
from onnx import TensorProto
from onnx import ValueInfoProto
from onnx.helper import make_graph
from onnx.helper import make_tensor
from onnx.helper import make_tensor_value_info
from onnx.helper import mapping
import tensorflow as tf
from tensorflow.core.framework.attr_value_pb2 import AttrValue
from tensorflow.core.framework.node_def_pb2 import NodeDef

from onnx_tf.common import attr_converter
from onnx_tf.common import attr_translator
from onnx_tf.common import IS_PYTHON3


class TensorflowNode(object):

  def __init__(self, node=None):
    # storing a reference to the original protobuf object
    if node is None:
      self.node = None
      self.name = ""
      self.inputs = []
      self.attr = {}
      self.domain = ""
      self.op_type = ""
    elif isinstance(node, (OnnxNode, NodeProto)):
      self._load_onnx_node(node)
    elif isinstance(node, NodeDef):
      self._load_tf_node(node)

  def _load_onnx_node(self, node):
    if isinstance(node, NodeProto):
      node = OnnxNode(node)
    self.name = node.name
    self.inputs = node.inputs
    self.attr = node.attrs
    self.domain = node.domain
    self.op_type = node.op_type

  def _load_tf_node(self, node):
    self.node = node
    self.name = node.name
    self.inputs = list(node.input)
    self.attr = {}
    for key, val in node.attr.items():
      new_val = attr_translator.translate_tf(key, val)
      if isinstance(new_val, AttrValue):
        new_val = attr_converter.convert_tf(new_val)
      self.attr[key] = new_val
    splitted_op_name = node.op.split(".")
    self.domain = "" if len(splitted_op_name) == 1 else ".".join(
        splitted_op_name[:-1])
    self.op_type = splitted_op_name[-1]


class TensorflowGraph(object):

  def __init__(self, graph_def, outputs=(), graph_name="graph"):
    self._graph_name = graph_name
    self._nodes = self._parse_nodes(graph_def.node)
    self._nodes_dict = {n.name: n for n in self._nodes}
    self._outputs = outputs or self.get_output_node_names(graph_def)
    self._graph_def = self._process_graph_def(graph_def)

  def get_node_by_name(self, name):
    node = self._nodes_dict.get(name, None)
    if node is None:
      raise ValueError(
          "Node {} is not found in the graph provided".format(name))
    return node

  def _process_graph_def(self, graph_def):
    if self._outputs and "_output_shapes" not in self.get_node_by_name(
        self._outputs[0]).attr:
      graph_def = self._add_infer_shapes(graph_def)
    return graph_def

  @staticmethod
  def _add_infer_shapes(graph_def):
    with tf.Graph().as_default():
      with tf.Session(
          config=tf.ConfigProto(
              graph_options=tf.GraphOptions(infer_shapes=True))) as sess:
        tf.import_graph_def(graph_def, name="")
      return sess.graph_def

  @staticmethod
  def get_output_node_names(graph_def):
    """Get output node names from GraphDef.

    Args:
      graph_def: GraphDef object.

    Returns:
      List of output node names.
    """
    input_names, output_names = set(), set()
    for node in graph_def.node:
      output_names.add(node.name)
      input_names.update(set(node.input))
    return list(output_names - input_names)

  @staticmethod
  def _parse_nodes(nodes):
    from onnx_tf.graph_parser import MultiRNNParser
    for parser in [MultiRNNParser]:
      nodes = parser.parse(nodes)
    return nodes

  @property
  def graph_def(self):
    return self._graph_def

  @property
  def graph_name(self):
    return self._graph_name

  @property
  def nodes(self):
    return self._nodes

  @property
  def nodes_dict(self):
    return self._nodes_dict

  @property
  def outputs(self):
    return self._outputs


# TODO: Move this into ONNX main library
class OnnxNode(object):
  """
  Reimplementation of NodeProto from ONNX, but in a form
  more convenient to work with from Python.
  """

  def __init__(self, node):
    self.name = str(node.name)
    self.op_type = str(node.op_type)
    self.domain = str(node.domain)
    self.attrs = dict([(attr.name,
                        attr_translator.translate_onnx(
                            attr.name, attr_converter.convert_onnx(attr)))
                       for attr in node.attribute])
    self.inputs = list(node.input)
    self.outputs = list(node.output)
    self.node_proto = node


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
