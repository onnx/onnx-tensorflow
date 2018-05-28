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
  def handle_g_r_u(cls, node, input_dict):
    if node.attrs.get("linear_before_reset", 0) != 0:
      raise NotImplementedError("linear_before_reset is not supported.")
    return TensorflowBackendV1.handle_g_r_u(node, input_dict)
