#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx_tf.third_party import get_info

import onnx_tf.backend
import onnx_tf.frontend

def main():
    gen_doc_for = {
        'onnx_tf.backend.TensorflowBackendBase': [
            onnx_tf.backend.TensorflowBackendBase.prepare,
            ],
        'onnx_tf.frontend.TensorflowFrontendBase': [
            onnx_tf.frontend.TensorflowFrontendBase.tensorflow_graph_to_onnx_model,
        ]
    }

    with open('./doc/API.md', 'w') as doc_file:
        doc_file.write('ONNX-Tensorflow API\n')
        doc_file.write('------\n\n')

        for scope, funcs in gen_doc_for.items():
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

if __name__ == '__main__':
    main()
