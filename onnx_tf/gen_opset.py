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
)

def main():
    opset_dict = {}

    for schema in defs.get_all_schemas_with_history():
        op_name = op_name_to_lower(schema.name)
        opset_dict[op_name] = []

    version = 1
    while True:
        try:
            backend = (importlib.import_module('backend_v{}'
                        .format(version))
                        .TensorflowBackend)
        except:
            break

        for schema in defs.get_all_schemas():
            op_name = op_name_to_lower(schema.name)
            if hasattr(backend, 'handle_' + op_name):
                opset_dict[op_name].append(version)
        version += 1

    with open('opset_version.py', 'w') as version_file:
        pp = pprint.PrettyPrinter(indent=4)
        version_file.write("opset_version = " + pp.pformat(opset_dict))


if __name__ == '__main__':
    main()
