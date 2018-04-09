#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals

import importlib
import pprint

from onnx import defs

from onnx_tf.common import (op_name_to_lower, ONNX_OP_TO_TF_OP,
                            ONNX_OP_TO_TF_OP_STR)


def main():
  backend_opset_dict = {}
  frontend_opset_dict = {}
  frontend_tf_opset_dict = {}

  for schema in defs.get_all_schemas():
    op_name = schema.name
    backend_opset_dict[op_name] = []
    frontend_opset_dict[op_name] = []

  version = 1
  while True:
    try:
      backend = (importlib.import_module('backends.backend_v{}'.format(version))
                 .TensorflowBackend)
      frontend = (importlib.import_module('frontends.frontend_v{}'
                                          .format(version)).TensorflowFrontend)
    except:
      break

    # Register all tf ops in ONNX_TO_HANDLER
    tf_op_names = []
    onnx_to_handler = frontend.ONNX_TO_HANDLER.get(
        'frontend_v{}'.format(version), {})
    # for handler in frontend.ONNX_TO_HANDLER.values():
    for handler in onnx_to_handler.values():
      if isinstance(handler, list):
        tf_op_names.extend(list(map(op_name_to_lower, handler)))
      else:
        tf_op_names.append(op_name_to_lower(handler))

    for schema in defs.get_all_schemas():
      op_name = schema.name
      lower_op_name = op_name_to_lower(op_name)
      has_backend_handler = hasattr(backend, 'handle_' + lower_op_name)
      # Record only one version for trivial ops
      if has_backend_handler or (version == 1 and
                                 lower_op_name in ONNX_OP_TO_TF_OP.keys()):
        backend_opset_dict[op_name].append(version)

      # Register once if onnx op in ONNX_OP_TO_TF_OP_STR
      if version == 1 and schema.name in ONNX_OP_TO_TF_OP_STR and \
          ONNX_OP_TO_TF_OP_STR[schema.name] not in tf_op_names:
        tf_op_names.append(op_name_to_lower(ONNX_OP_TO_TF_OP_STR[schema.name]))
        frontend_opset_dict[op_name].append(version)
      # Register if onnx op in ONNX_TO_HANDLER
      elif op_name in onnx_to_handler:
        frontend_opset_dict[op_name].append(version)
    for tf_op_name in tf_op_names:
      frontend_tf_opset_dict.setdefault(str(tf_op_name), []).append(version)

    version += 1

  with open('opset_version.py', 'w') as version_file:
    pp = pprint.PrettyPrinter(indent=4)
    version_file.write("backend_opset_version = {\n " +
                       pp.pformat(backend_opset_dict)[1:-1] + "\n}\n\n")
    version_file.write("frontend_opset_version = {\n " +
                       pp.pformat(frontend_opset_dict)[1:-1] + "\n}\n\n")
    version_file.write("frontend_tf_opset_version = {\n " +
                       pp.pformat(frontend_tf_opset_dict)[1:-1] + "\n}\n")


if __name__ == '__main__':
  main()
