#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx_tf.third_party import get_info

import onnx_tf.backend
import onnx_tf.frontend

import onnx_tf.backend_rep
import opset_version


def main():
  gen_doc_for = {
      'onnx_tf.backend': [
          onnx_tf.backend.prepare,
      ],
      'onnx_tf.frontend': [
          onnx_tf.frontend.tensorflow_graph_to_onnx_model,
      ],
      'onnx_tf.backend_rep.TensorflowRep': [
          onnx_tf.backend_rep.TensorflowRep.export_graph,
      ]
  }

  with open('./doc/API.md', 'w') as doc_file:
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

  with open('./doc/support_status.md', 'w') as status_file:
    status_file.write('ONNX-Tensorflow Support Status\n')
    status_file.write('======\n\n')

    status_file.write('Backend\n')
    status_file.write('______\n\n')

    status_file.write('| ONNX Op        | Supported ONNX Version  |\n')
    status_file.write('| -------------- |:------------------:|\n')
    for key, val in sorted(opset_version.backend_opset_version.items()):
      version_str = str(val)[1:-1]
      status_file.write("|{}|{}|\n".format(key, version_str
                                           if len(version_str) else "N/A"))

    status_file.write('\n\n')

    status_file.write('Frontend\n')
    status_file.write('______\n\n')

    status_file.write('| Tensorflow Op        | Supported ONNX Version  |\n')
    status_file.write('| -------------- |:------------------:|\n')
    for key, val in sorted(opset_version.frontend_tf_opset_version.items()):
      version_str = str(val)[1:-1]
      status_file.write("|{}|{}|\n".format(key, version_str
                                           if len(version_str) else "N/A"))


if __name__ == '__main__':
  main()
