#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import subprocess

import onnx_tf.backend
import onnx_tf.backend_rep
from third_party import get_info


def main(docs_dir):
  gen_api(docs_dir)
  gen_cli(docs_dir)


def gen_api(docs_dir):
  gen_doc_for = {
      'onnx_tf.backend': [
          onnx_tf.backend.prepare,
      ],
      'onnx_tf.backend_rep.TensorflowRep': [
          onnx_tf.backend_rep.TensorflowRep.export_graph,
      ]
  }
  with open(os.path.join(docs_dir, 'API.md'), 'w') as doc_file:
    doc_file.write('ONNX-Tensorflow API\n')
    doc_file.write('======\n\n')

    for scope, funcs in sorted(gen_doc_for.items()):
      for func in funcs:
        doc_parsed = get_info.parse_docstring(func.__doc__)
        doc_file.write('#### `' + scope + '.' + func.__name__ + '`\n\n')
        doc_file.write('<details>\n')
        doc_file.write('  <summary>')
        doc_file.write(doc_parsed['short_description'] + '\n\n')
        doc_file.write('  </summary>\n')
        doc_file.write(doc_parsed['long_description'] + '\n\n')
        doc_file.write('</details>\n\n\n\n')

        doc_file.write('_params_:\n\n')
        for param in doc_parsed['params']:
          doc_file.write('`' + param['name'] + '` : ' + param['doc'] + '\n\n')

        doc_file.write('_returns_:\n\n')
        doc_file.write(doc_parsed['returns'] + '\n\n')


def gen_cli(docs_dir):
  with open(os.path.join(docs_dir, 'CLI_template.md'), 'r') as cli_temp_file:
    temp_lines = cli_temp_file.readlines()

  lines = []
  for line in temp_lines:
    matched = re.match(r"{onnx-tf.*}", line)
    if matched:
      command = matched.string.strip()[1:-1]
      output = subprocess.check_output(command.split(" ")).decode("UTF-8")
      lines.append(output)
    else:
      lines.append(line)

  with open(os.path.join(docs_dir, 'CLI.md'), 'w') as cli_file:
    cli_file.writelines(lines)


if __name__ == '__main__':
  base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
  docs_dir = os.path.join(base_dir, 'doc')
  main(docs_dir)
