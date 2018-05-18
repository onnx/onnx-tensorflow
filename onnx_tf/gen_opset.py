#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals

import importlib
import pprint

from onnx import defs

from onnx_tf.common import op_name_to_lower, ONNX_OP_TO_TF_OP
from onnx_tf.common.handler_helper import get_frontend_coverage


def main():
  backend_opset_dict = {}
  frontend_opset_dict = {}

  for schema in defs.get_all_schemas():
    op_name = schema.name
    backend_opset_dict[op_name] = []
    frontend_opset_dict[op_name] = []

  version = 1
  while True:
    try:
      backend = (importlib.import_module('backends.backend_v{}'.format(version))
                 .TensorflowBackend)
    except:
      break

    for schema in defs.get_all_schemas():
      op_name = schema.name
      lower_op_name = op_name_to_lower(op_name)
      has_backend_handler = hasattr(backend, 'handle_' + lower_op_name)
      # Record only one version for trivial ops
      if has_backend_handler or (version == 1 and
                                 lower_op_name in ONNX_OP_TO_TF_OP.keys()):
        backend_opset_dict[op_name].append(version)

    version += 1

  frontend_onnx_coverage, frontend_tf_coverage = get_frontend_coverage()
  frontend_opset_dict.update(frontend_onnx_coverage)

  with open('opset_version.py', 'w') as version_file:
    pp = pprint.PrettyPrinter(indent=4)
    version_file.write("backend_opset_version = {\n " +
                       pp.pformat(backend_opset_dict)[1:-1] + "\n}\n\n")
    version_file.write("frontend_opset_version = {\n " +
                       pp.pformat(frontend_opset_dict)[1:-1] + "\n}\n\n")
    version_file.write("frontend_tf_opset_version = {\n " +
                       pp.pformat(frontend_tf_coverage)[1:-1] + "\n}\n")


if __name__ == '__main__':
  main()
