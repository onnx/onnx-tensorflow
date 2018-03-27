"""Backend for running ONNX on Tensorflow

To run this, you will need to have Tensorflow installed as well.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from onnx_tf.backend import TensorflowBackendBase


class TensorflowBackend(TensorflowBackendBase):
  """ Tensorflow Backend for ONNX
  """

  @classmethod
  def handle_reshape(cls, node, input_dict):
    tensor = input_dict[node.inputs[0]]
    shape = input_dict[node.inputs[1]]
    return [tf.reshape(tensor, shape)]
