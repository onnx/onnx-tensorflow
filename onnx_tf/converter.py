import argparse
import inspect
import logging
import os
import shutil

import onnx
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.tools import freeze_graph

import onnx_tf.backend as backend
from onnx_tf.common import get_output_node_names
from onnx_tf.common import get_unique_suffix
import onnx_tf.frontend as frontend

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


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
        if value.isdigit():
          res.append(int(value))
        else:
          res.append(value)
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
  parser.add_argument(
      "--infile",
      "-i",
      help="Input file path, can be pb or ckpt file.",
      required=True)
  parser.add_argument(
      "--outfile", "-o", help="Output file path.", required=True)
  parser.add_argument(
      "--convert_to",
      "-t",
      choices=["onnx", "tf"],
      help="Format converted to.",
      required=True)
  parser.add_argument(
      "--graph",
      "-g",
      help=
      "Inference graph, which is obtained by optimizing or editing the training graph for better training usability."
  )

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
                         "strict": {}
                     })])

  # frontend args
  # Args must be named consistently with respect to frontend.tensorflow_graph_to_onnx_model.
  add_argument_group(parser, "frontend arguments (tf -> onnx)",
                     [(frontend.tensorflow_graph_to_onnx_model, {
                         "output": {
                             "action": ListAction,
                             "dest": "output"
                         },
                         "opset": {
                             "action": OpsetAction,
                         },
                         "ignore_unimplemented": {
                             "type": bool
                         },
                         "optimizer_passes": {
                             "action": ListAction,
                             "dest": "optimizer_passes"
                         }
                     })])

  return parser.parse_args(args)


def convert(infile, outfile, convert_to, graph=None, **kwargs):
  """Convert pb.

  Args:
    infile: Input path.
    outfile: Output path.
    convert_to: Format converted to.
    graph: Inference graph.
    **kwargs: Other args for converting.

  Returns:
    None.
  """
  if convert_to == "tf":
    logger.info("Start converting onnx pb to tf pb:")
    onnx_model = onnx.load(infile)
    tf_rep = backend.prepare(onnx_model, **kwargs)
    tf_rep.export_graph(outfile)
  elif convert_to == "onnx":
    ext = os.path.splitext(infile)[1]
    logger.info("Start converting tf pb to onnx pb:")
    if ext == ".pb":
      with open(infile, "rb") as f:
        graph_def = graph_pb2.GraphDef()
        graph_def.ParseFromString(f.read())
    elif ext == ".ckpt":
      latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(infile))
      saver = tf.train.import_meta_graph(latest_ckpt + ".meta")
      output_node_names = []
      temp_file_suffix = get_unique_suffix()
      workdir = 'onnx-tf_workdir_{}'.format(temp_file_suffix)
      with tf.Session() as sess:
        sess.run([
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        ])
        saver.restore(sess, latest_ckpt)
        # Take users' hint or deduce output node automatically.
        kwargs["output"] = kwargs.get("output", None) or get_output_node_names(
            sess.graph.as_graph_def())

        # Save the graph to disk for freezing.
        tf.train.write_graph(
            sess.graph.as_graph_def(add_shapes=True),
            workdir,
            "input_model.pb",
            as_text=False)

      # Freeze graph:
      freeze_graph.freeze_graph(
          input_graph=graph or workdir + "/input_model.pb",
          input_saver="",
          input_binary=True,
          input_checkpoint=latest_ckpt,
          output_node_names=",".join(kwargs["output"]),
          restore_op_name="",
          filename_tensor_name="",
          output_graph=workdir + "/frozen_model.pb",
          clear_devices=True,
          initializer_nodes="")

      # Load back the frozen graph.
      with open(workdir + "/frozen_model.pb", "rb") as f:
        graph_def = graph_pb2.GraphDef()
        graph_def.ParseFromString(f.read())

      # Remove work directory.
      shutil.rmtree(workdir)
    else:
      raise ValueError(
          "Input file is not supported. Should be .pb or .ckpt, but get {}".
          format(ext))
    onnx_model = frontend.tensorflow_graph_to_onnx_model(graph_def, **kwargs)
    onnx.save(onnx_model, outfile)
  logger.info("Converting completes successfully.")
