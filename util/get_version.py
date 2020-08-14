from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import onnx
import tensorflow
import onnx_tf

print("Python version:")
print(sys.version)
print("ONNX version:")
print(onnx.version.version)
print("ONNX-TF version:")
print(onnx_tf.__version__)
print("Tensorflow version:")
print(tensorflow.__version__)
