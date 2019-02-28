#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pprint

from onnx import defs

from onnx_tf.common.handler_helper import get_frontend_coverage
from onnx_tf.common.handler_helper import get_backend_coverage


def main():
  backend_opset_dict = {}
  frontend_opset_dict = {}

  for schema in defs.get_all_schemas():
    op_name = schema.name
    backend_opset_dict[op_name] = []
    frontend_opset_dict[op_name] = []

  backend_onnx_coverage, backend_experimental_op = get_backend_coverage()
  backend_opset_dict.update(backend_onnx_coverage.get(defs.ONNX_DOMAIN, {}))
  frontend_coverages = get_frontend_coverage()
  frontend_onnx_coverage = frontend_coverages.get('onnx_coverage')
  frontend_tf_coverage = frontend_coverages.get('tf_coverage')
  experimental_op = frontend_coverages.get('experimental_op')
  frontend_opset_dict.update(frontend_onnx_coverage.get(defs.ONNX_DOMAIN, {}))

  for exp_op in experimental_op:
    frontend_opset_dict["{} [EXPERIMENTAL]".format(
        exp_op)] = frontend_opset_dict.pop(exp_op)

  with open('opset_version.py', 'w') as version_file:
    pp = pprint.PrettyPrinter(indent=4)
    version_file.write("backend_opset_version = {\n " +
                       pp.pformat(backend_opset_dict)[1:-1] + "\n}\n\n")
    version_file.write("frontend_opset_version = {\n " +
                       pp.pformat(frontend_opset_dict)[1:-1] + "\n}\n\n")
    version_file.write("frontend_tf_opset_version = {\n " + pp.pformat(
        frontend_tf_coverage.get(defs.ONNX_DOMAIN, {}))[1:-1] + "\n}\n")


if __name__ == '__main__':
  main()
