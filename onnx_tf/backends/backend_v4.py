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
  def handle_concat(cls, node, input_dict):
    values = [input_dict[a] for a in node.inputs]
    assert "axis" in node.attrs, ("axis is required.")
    axis = node.attrs["axis"]
    return [tf.concat(values, axis=axis)]
