"""Frontend for exporting Tensorflow graph to ONNX graph

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx_tf.frontend import TensorflowFrontendBase


class TensorflowFrontend(TensorflowFrontendBase):
  """ Tensorflow Frontend for ONNX
  """
