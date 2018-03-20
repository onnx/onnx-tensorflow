#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import defs
from onnx.defs import OpSchema

import importlib
import pprint

from onnx_tf.common import (
  op_name_to_lower,
  ONNX_OP_TO_TF_OP,
  ONNX_OP_TO_TF_OP_STR
)

def main():
    backend_opset_dict = {}
    frontend_opset_dict = {}
    frontend_tf_opset_dict = {}

    for schema in defs.get_all_schemas():
        op_name = op_name_to_lower(schema.name)
        backend_opset_dict[op_name] = []
        frontend_opset_dict[op_name] = []

    version = 1
    while True:
        try:
            backend = (importlib.import_module('backends.backend_v{}'
                                .format(version))
                                .TensorflowBackend)
            frontend = (importlib.import_module('frontends.frontend_v{}'
                                .format(version))
                                .TensorflowFrontend)
        except:
            break

        for schema in defs.get_all_schemas():
            op_name = op_name_to_lower(schema.name)
            has_backend_handler = hasattr(backend, 'handle_' + op_name)
            if has_backend_handler or (op_name in ONNX_OP_TO_TF_OP.keys()):
                backend_opset_dict[op_name].append(version)

            if op_name in frontend.ONNX_TO_HANDLER:
                tf_op_name = op_name_to_lower(frontend.ONNX_TO_HANDLER[op_name])
            elif schema.name in ONNX_OP_TO_TF_OP_STR.keys():
                tf_op_name = op_name_to_lower(ONNX_OP_TO_TF_OP_STR[schema.name])
            else:
                continue
            if tf_op_name in frontend_tf_opset_dict:
                frontend_tf_opset_dict[tf_op_name].append(version)
            else:
                frontend_tf_opset_dict[tf_op_name] = [version]
            frontend_opset_dict[op_name].append(version)

        version += 1

    with open('opset_version.py', 'w') as version_file:
        pp = pprint.PrettyPrinter(indent=4)
        version_file.write("backend_opset_version = {\n " + pp.pformat(backend_opset_dict)[1:] + "\n\n")
        version_file.write("frontend_opset_version = {\n " + pp.pformat(frontend_opset_dict)[1:] + "\n\n")
        version_file.write("frontend_tf_opset_version = {\n " + pp.pformat(frontend_tf_opset_dict)[1:] + "\n")

if __name__ == '__main__':
    main()
