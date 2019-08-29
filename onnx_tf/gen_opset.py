#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pprint

from onnx import defs

from onnx_tf.common.handler_helper import get_backend_coverage


def main():
  backend_opset_dict = {}

  for schema in defs.get_all_schemas():
    op_name = schema.name
    backend_opset_dict[op_name] = []

  backend_onnx_coverage, backend_experimental_op = get_backend_coverage()
  backend_opset_dict.update(backend_onnx_coverage.get(defs.ONNX_DOMAIN, {}))

  with open('opset_version.py', 'w') as version_file:
    pp = pprint.PrettyPrinter(indent=4)
    version_file.write("backend_opset_version = {\n " +
                       pp.pformat(backend_opset_dict)[1:-1] + "\n}\n\n")


if __name__ == '__main__':
  main()
