"""Backend for running ONNX on Tensorflow

To run this, you will need to have Tensorflow installed as well.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx_tf.backend import TensorflowBackendBase
from onnx_tf.backends.backend_v1 import TensorflowBackend as TensorflowBackendV1

class TensorflowBackend(TensorflowBackendBase):
  """ Tensorflow Backend for ONNX
  """

  @classmethod
  def handle_batch_normalization(cls, node, input_dict):
    return TensorflowBackendV1.handle_batch_normalization(node, input_dict)
