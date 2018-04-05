"""Frontend for exporting Tensorflow graph to ONNX graph

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import importlib
from itertools import chain

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.tensor_util import MakeNdarray
from tensorflow.core.framework.attr_value_pb2 import AttrValue

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

      if isinstance(self.attr[new_key], AttrValue):
        self.attr[new_key] = get_attribute_value(self.attr[new_key])

  def type_converter(self, x):
    return TF_TYPE_TO_ONNX_TYPE[tf.as_dtype(x.type)]


class TensorflowFrontendBase(object):
  """ Tensorflow Frontend for ONNX
  """

  DEFAULT_TF_ATTR_PER_OP = {
      "Add": {
          "broadcast": 1
      },
      "And": {
          "broadcast": 1
      },
      "Div": {
          "broadcast": 1
      },
      "Equal": {
          "broadcast": 1
      },
      "Greater": {
          "broadcast": 1
      },
      "Less": {
          "broadcast": 1
      },
      "Mul": {
          "broadcast": 1
      },
      "Or": {
          "broadcast": 1
      },
      "Pow": {
          "broadcast": 1
      },
      "Sub": {
          "broadcast": 1
      },
      "Xor": {
          "broadcast": 1
      },
  }

  frontend_version_cache = {}

  @classmethod
  def tensorflow_graph_to_onnx_graph(cls,
                                     graph_def,
                                     output,
                                     opset=(("", 0),),
                                     name="graph"):
    """Converts a Tensorflow Graph Proto to an ONNX graph

    This function converts a Tensorflow Graph proto to an equivalent
    representation of ONNX graph.

    :param graph_def: Tensorflow Graph Proto object.
    :param output: A Tensorflow NodeDef object specifying which node
      to be taken as output of the ONNX graph.
    :param opset: Opset, which should be ((str domain: int version number),).
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
        splitted_op_name = node.op.split(".")
        op_domain = "" if len(splitted_op_name) == 1 else ".".join(
            splitted_op_name[:-1])
        op_name = splitted_op_name[-1]

        handler_name = "handle_" + op_name_to_lower(op_name)

        # TODO per domain frontend_tf_opset_version?
        versions = frontend_tf_opset_version[op_name_to_lower(op_name)]

        opset_dict = {}
        onnx_domain = defs.ONNX_DOMAIN
        for domain, version in opset:
          if domain == "ai.onnx":
            domain = ""
          opset_dict[domain] = version
          defs.ONNX_DOMAIN = domain
          assert isinstance(
              version, int
          ) and (version <= defs.onnx_opset_version()) and (
              version >= 0
          ), "Opset should be an int less than or equal to {}, but {}: {}".format(
              defs.onnx_opset_version(), type(version), version)
          defs.ONNX_DOMAIN = onnx_domain

        opset_ver = opset_dict[op_domain]
        if opset_ver == 0:
          version = max(versions)
        else:
          versions = sorted(versions + [opset_ver])
          version = versions[
              max([i for i, v in enumerate(versions) if v == opset_ver]) - 1]

        camel_domain = "".join(w.title() for w in op_domain.split("."))
        frontend_ver = "frontend_v{}".format(version)
        frontend_class_name = "{}TensorflowFrontend".format(camel_domain)
        frontend_module = cls.frontend_version_cache.setdefault(
            frontend_ver,
            importlib.import_module("onnx_tf.frontends." + frontend_ver))
        if hasattr(frontend_module, frontend_class_name):
          frontend = getattr(frontend_module, frontend_class_name)
        else:
          assert NotImplementedError, \
            "{} for domain {} is not implemented".format(frontend_ver, op_domain)

        # Check if specialized handler exists.
        if hasattr(frontend, handler_name):
          method_to_call = getattr(frontend, handler_name)
          node = method_to_call(node, consts=consts, node_dict=dict(node_tup))
          if isinstance(node, list):
            ops_proto.extend(node)
          else:
            ops_proto.append(node)
        elif node.op in TF_OP_STR_TO_ONNX_OP.keys():
          # Remove tensorflow-specific attrs that are not
          # needed/allowed in ONNX.
          attr = cls.DEFAULT_TF_ATTR_PER_OP.get(node.op, {})
          filtered_attr = dict(
                            filter(lambda pair: pair[0] not in TF_ATTR_TO_REMOVE,
                                   node.attr.items()))
          node_output = name
          ops_proto.append(
              make_node(
                  TF_OP_STR_TO_ONNX_OP[node.op],
                  node.inputs, [node_output],
                  name=name,
                  **filtered_attr))
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
      output_proto.append(
          make_tensor_value_info(output_name, output_onnx_type,
                                 output.attr["_output_shapes"][i]))

    inputs = list(chain.from_iterable(map(lambda p: list(p.input), ops_proto)))

    # Remove proto in inputs_proto and consts_proto if proto is not used as input in ONNX
    inputs_proto = list(filter(lambda x: x.name in inputs, inputs_proto))
    consts_proto = list(filter(lambda x: x.name in inputs, consts_proto))

    return make_graph(ops_proto, name, inputs_proto, output_proto, consts_proto)

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
    :param output: A string specifying the name of the output
      graph node.
    :param opset: Opset version number, list or tuple.
      Default is 0 means using latest version with domain ''.
      List or tuple items should be (str domain, int version number).
    :param producer_name: The name of the producer.
    :param graph_name: The name of the output ONNX Graph.

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
        (int, list,
         tuple)), "opset is expected to int, list or tuple, but {}.".format(
             type(opset))
    if isinstance(opset, int):
      opset = [("", opset)]
    opset_imports = [helper.make_opsetid(item[0], item[1]) for item in opset]

    output_node = get_node_by_name(graph_def.node, output)
    onnx_graph = cls.tensorflow_graph_to_onnx_graph(graph_def, output_node,
                                                    opset, graph_name)
    onnx_model = make_model(
        onnx_graph, producer_name=producer_name, opset_imports=opset_imports)

    return onnx_model

  @classmethod
  def _bin_op(cls, node, onnx_op, axis=None):
    node.attr["broadcast"] = 1
    if (axis):
      return helper.make_node(
          onnx_op, node.inputs, [node.name], name=node.name, broadcast=1, axis=axis)
    else:
      return helper.make_node(
          onnx_op, node.inputs, [node.name], name=node.name, broadcast=1)

  @classmethod
  def _pool_op(cls, node, onnx_op, **kwargs):
    auto_pad = node.attr["padding"].decode("UTF-8")
    auto_pad = "SAME_UPPER" if auto_pad == "SAME" else auto_pad
    data_format = node.attr["data_format"].decode("UTF-8")
    spatial_indices = [
        i for i in range(len(data_format)) if data_format[i] not in ["N", "C"]
    ]
    strides = list(map(lambda i: node.attr["strides"][i], spatial_indices))
    kernel_shape = list(map(lambda i: node.attr["ksize"][i], spatial_indices))
    node_dict = kwargs["node_dict"]
    output_shape = list(
        map(lambda i: node.attr["_output_shapes"][0][i], spatial_indices))
    input_shape = list(
        map(lambda i: node_dict[node.inputs[0]].attr["_output_shapes"][0][i], spatial_indices))
    pads = cls._cal_pads(auto_pad, len(spatial_indices), input_shape,
                         output_shape, strides, kernel_shape)
    return helper.make_node(
        onnx_op, [node.inputs[0]], [node.name],
        pads=pads,
        kernel_shape=kernel_shape,
        strides=strides)

  @classmethod
  def _cal_pads(cls, auto_pad, spatial_dim, input_shape, output_shape, strides,
                kernel_shape):
    pads = [0] * spatial_dim * 2
    if auto_pad == "SAME_UPPER":
      for i in range(spatial_dim):
        pad_shape = (
            output_shape[i] - 1) * strides[i] + kernel_shape[i] - input_shape[i]
        pads[i] = pad_shape // 2
        pads[i + spatial_dim] = pad_shape - pad_shape // 2
    return pads

  @classmethod
  def _reduce_op(cls, op, node, **kwargs):
    consts = kwargs["consts"]
    assert node.inputs[1] in consts.keys()
    axes = consts[node.inputs[1]]
    return helper.make_node(
        op, [node.inputs[0]], [node.name],
        axes=axes,
        keepdims=node.attr.get("keep_dims", 1))


convert_graph = TensorflowFrontendBase.tensorflow_graph_to_onnx_graph

tensorflow_graph_to_onnx_model = TensorflowFrontendBase.tensorflow_graph_to_onnx_model
