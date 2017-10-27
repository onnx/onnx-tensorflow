from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx_tf.backend import run_node, prepare
from onnx import helper

node_def = helper.make_node("Relu", ["X"], ["Y"])
output = run_node(node_def, [[-0.1, 0.1]])
print(output["Y"])
