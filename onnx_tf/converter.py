import argparse
import inspect
import logging
import os
import shutil

import onnx
from onnx.external_data_helper import load_external_data_for_model
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.tools import freeze_graph

import onnx_tf.backend as backend
import onnx_tf.common as common
from onnx_tf.common import get_unique_suffix
from onnx_tf.pb_wrapper import TensorflowGraph


def main(args):
  args = parse_args(args)
  convert(**{k: v for k, v in vars(args).items() if v is not None})


def parse_args(args):

  class ListAction(argparse.Action):
    """ Define how to convert command line list strings to Python objects.
    """

    def __call__(self, parser, namespace, values, option_string=None):
      values = values if values[0] not in ("(", "[") or values[-1] not in (
          ")", "]") else values[1:-1]
      res = []
      for value in values.split(","):
        res.append(int(value) if value.isdigit() else value)
      setattr(namespace, self.dest, res)

  class OpsetAction(argparse.Action):
    """ Define how to convert command line opset strings to Python objects.
    """

    def __call__(self, parser, namespace, values, option_string=None):
      if values.isdigit():
        setattr(namespace, "opset", int(values))
      else:
        res = []
        while values and values[0] in ("(", "["):
          values = values[1:]
        while values and values[-1] in (")", "]"):
          values = values[:-1]
        for value in values.split("),("):
          l, r = value.split(",")
          res.append((l, int(r)))
        setattr(namespace, "opset", res)

  def get_param_doc_dict(funcs):
    """Get doc of funcs params.

    Args:
      funcs: Target funcs.

    Returns:
      Dict of params doc.
    """

    # TODO(fumihwh): support google doc format
    def helper(doc, func):
      first_idx = doc.find(":param")
      last_idx = doc.find(":return")
      last_idx = last_idx if last_idx != -1 else len(doc)
      param_doc = doc[first_idx:last_idx]
      params_doc = param_doc.split(":param ")[1:]
      return {
          p[:p.find(": ")]: p[p.find(": ") + len(": "):] +
          " (from {})".format(func.__module__ + "." + func.__name__)
          for p in params_doc
      }

    param_doc_dict = {}
    for func, persists in funcs:
      doc = inspect.getdoc(func)
      doc_dict = helper(doc, func)
      for k, v in doc_dict.items():
        if k not in persists:
          continue
        param_doc_dict[k] = {"doc": v, "params": persists[k]}
    return param_doc_dict

  parser = argparse.ArgumentParser(
      description=
      "This is the converter for converting protocol buffer between tf and onnx."
  )

  # required two args, source and destination path
  parser.add_argument("--infile", "-i", help="Input file path.", required=True)
  parser.add_argument("--outdir", "-o", help="Output directory.", required=True)
  parser.add_argument("--extdatadir",
                      "-e",
                      help="External input data file directory.",
                      required=False)

  def add_argument_group(parser, group_name, funcs):
    group = parser.add_argument_group(group_name)
    param_doc_dict = get_param_doc_dict(funcs)
    for k, v in param_doc_dict.items():
      group.add_argument("--{}".format(k), help=v["doc"], **v["params"])

  # backend args
  # Args must be named consistently with respect to backend.prepare.
  add_argument_group(parser, "backend arguments (onnx -> tf)",
                     [(backend.prepare, {
                         "device": {},
                         "strict": {},
                         "logging_level": {},
                         "auto_cast": {}
                     })])

  return parser.parse_args(args)


def convert(infile, outdir, **kwargs):
  """Convert pb.

  Args:
    infile: Input path.
    outdir: Output path.
    **kwargs: Other args for converting.

  Returns:
    None.
  """
  logging_level = kwargs.get("logging_level", "INFO")
  ext_data_dir = kwargs.get("extdatadir")

  common.logger.setLevel(logging_level)
  common.logger.handlers[0].setLevel(logging_level)
  common.logger.info("Start converting onnx pb to tf saved model")

  # load external data if the file directory is provided
  if ext_data_dir:
    onnx_model = onnx.load(infile, load_external_data=False)
    load_external_data_for_model(onnx_model, ext_data_dir)
  else:
    onnx_model = onnx.load(infile)

  tf_rep = backend.prepare(onnx_model, **kwargs)
  tf_rep.export_graph(outdir)
  common.logger.info("Converting completes successfully.")
