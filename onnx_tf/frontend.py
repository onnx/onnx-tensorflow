"""Frontend for exporting Tensorflow graph to ONNX graph

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from onnx import defs
from onnx.helper import make_model
from onnx.helper import make_opsetid
from onnx.optimizer import optimize
import tensorflow as tf

from onnx_tf.common import exception
from onnx_tf.common.handler_helper import get_all_frontend_handlers
from onnx_tf.common import IS_PYTHON3
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.pb_wrapper import TensorflowNode
from onnx_tf.pb_wrapper import OnnxGraph

# Define long type for Python 3:
if IS_PYTHON3:
  long = int

logger = logging.getLogger()


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
    training_ops_to_remove = ["RandomShuffleQueueV2"]

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
      elif node.op_type in training_ops_to_remove:
        logger.info(
            "A training op with name {} type {} has been removed.".format(
                node.name, node.op_type))
      elif node.op_type == "QueueDequeueManyV2":
        num_output = len(node.attr["_output_shapes"])
        for index, shape, onnx_type in zip(
            range(num_output), node.attr["_output_shapes"],
            node.attr["component_types"]):
          onnx_graph.add_input_proto_explicit(
              node.name + ":" + str(index), shape, onnx_dtype=onnx_type)
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
