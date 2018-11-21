from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import tensorflow as tf

import onnx
from onnx import mapping
from onnx import GraphProto

from onnx_tf.backend import run_node
from onnx_tf.pb_wrapper import OnnxGraph

logger = logging.getLogger()

def parse_args(args):
  # TODO: allow selective enablement of optimization passes
  parser = argparse.ArgumentParser(
      description="onnx-tensorflow optimization passes.")
  parser.add_argument(
      "--infile",
      "-i",
      help="Path to the ONNX model being optimized.",
      required=True)
  parser.add_argument(
      "--outfile",
      "-o",
      help="Output file path for the optimized ONNX model.",
      required=True)
  return parser.parse_args(args)


def constant_folding(onnx_graph):
  """ Remove constant nodes by evaluating them offline.
  """
  for node in onnx_graph.nodes_proto:
    # See if all inputs are present as contant tensors.
    inclusion_mask = map(lambda x: x in onnx_graph.consts, node.input)
    all_constant = all(inclusion_mask)
    # If all inputs are constant, then fold this constant node.
    if all_constant:
      logger.info("Folding a {} op with name {}".format(node.op_type, node.name))
      const_inputs = list(map(lambda x: onnx_graph.consts[x], node.input))
      outputs = run_node(node, const_inputs)
      # Make output tensors appear as graph initializers.
      for index, output_name in enumerate(node.output):
        output_content = outputs[index]
        output_onnx_type = mapping.NP_TYPE_TO_TENSOR_TYPE[output_content.dtype]
        onnx_graph.add_const_explicit(name=output_name, value=output_content)
        onnx_graph.add_const_proto_explicit(
            name=output_name, value=output_content, onnx_dtype=output_onnx_type)
        onnx_graph.add_input_proto_explicit(
            name=output_name,
            shape=output_content.shape,
            onnx_dtype=output_onnx_type)
      # Remove this folded constant node from graph.
      onnx_graph.remove_node_proto(node.name)
  return onnx_graph


all_optimization_passes = {"CONSTANT_FOLDING": constant_folding}

all_optimization_pass_names = all_optimization_passes.keys()


def optimize(onnx_model, passes=all_optimization_pass_names):
  """Optimize ONNX model (only graph for now).
  """
  onnx_graph = OnnxGraph(graph_proto=onnx_model.graph)
  for opt_pass in passes:
    assert opt_pass in all_optimization_passes.keys()
    opt_func = all_optimization_passes[opt_pass]
    onnx_graph = opt_func(onnx_graph)
  onnx_model.graph.CopyFrom(onnx_graph.make_graph_proto())
  return onnx_model


def main(args):
  args = parse_args(args)
  onnx_model = onnx.load(args.infile)
  onnx_model = optimize(onnx_model)
  onnx.save(onnx_model, args.outfile)
