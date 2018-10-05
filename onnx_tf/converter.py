import argparse
import inspect

import onnx
from tensorflow.core.framework import graph_pb2

import onnx_tf.backend as backend
import onnx_tf.frontend as frontend


def main(src, dest, **kwargs):
  onnx_model = onnx.load(src)
  if onnx_model.ir_version != 0:
    tf_rep = backend.prepare(onnx_model, **kwargs)
    tf_rep.export_graph(dest)
    return

  with open(src, "rb") as f:
    graph_def = graph_pb2.GraphDef()
    graph_def.ParseFromString(f.read())
  nodes, input_names = dict(), set()
  for node in graph_def.node:
    nodes[node.name] = node
    input_names.update(set(node.input))
  output = list(set(nodes) - input_names)
  onnx_model = frontend.tensorflow_graph_to_onnx_model(graph_def, output,
                                                       **kwargs)
  onnx.save(onnx_model, dest)


def parse_args():

  class ListAction(argparse.Action):

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

    def __call__(self, parser, namespace, values, option_string=None):
      if values.isdigit():
        setattr(namespace, "opset", int(values))
        return
      res = []
      while values and values[0] in ("(", "["):
        values = values[1:]
      while values and values[-1] in (")", "]"):
        values = values[:-1]
      for value in values.split("),("):
        l, r = value.split(",")
        res.append((l, int(r)))
      setattr(namespace, "opset", res)

  parser = argparse.ArgumentParser(
      description=
      "This is the converter for converting protocol buffer between onnx and tf."
  )
  parser.add_argument("--src", help="Path for model.")
  parser.add_argument("--dest", help="Path for exporting.")

  def get_param_doc_dict(funcs):

    def helper(doc, func):
      first_idx = doc.find(":param")
      last_idx = doc.find(":return")
      last_idx = last_idx if last_idx != -1 else len(doc)
      param_doc = doc[first_idx:last_idx]
      params_doc = param_doc.split(":param ")[1:]
      return {
          p[:p.find(": ")]:
          p[p.find(": ") + len(": "):] + " ({})".format(func.__name__)
          for p in params_doc
      }

    param_doc_dict = {}
    for func, persists in funcs:
      doc = inspect.getdoc(func)
      doc_dict = helper(doc, func)
      for k, v in doc_dict.items():
        if k not in persists:
          continue
        key = k if k not in param_doc_dict else k + "_{}".format(func.__name__)
        param_doc_dict[key] = {"doc": v, "params": persists[k]}
    return param_doc_dict

  backend_group = parser.add_argument_group("backend arguments (onnx -> tf)")
  backend_funcs = [(backend.prepare, {"device": {}, "strict": {}})]
  backend_param_doc_dict = get_param_doc_dict(backend_funcs)
  for k, v in backend_param_doc_dict.items():
    backend_group.add_argument("--{}".format(k), help=v["doc"], **v["params"])

  frontend_group = parser.add_argument_group("frontend arguments (tf -> onnx)")
  frontend_funcs = [(frontend.tensorflow_graph_to_onnx_model, {
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
  })]
  frontend_param_doc_dict = get_param_doc_dict(frontend_funcs)
  for k, v in frontend_param_doc_dict.items():
    frontend_group.add_argument("--{}".format(k), help=v["doc"], **v["params"])

  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  main(args.src, args.dest,
       **{k: v
          for k, v in vars(args).items()
          if v is not None})
