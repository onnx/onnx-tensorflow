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

  Lots of handler in this file just call backend_v1 due to
  ONNX removed `consumed_inputs` attr for them which is not used in handler.
  """

  @classmethod
  def handle_abs(cls, node, input_dict):
    return TensorflowBackendV1.handle_trivial(node, input_dict)

  @classmethod
  def handle_add(cls, node, input_dict):
    return TensorflowBackendV1.handle_add(node, input_dict)

  @classmethod
  def handle_batch_normalization(cls, node, input_dict):
    return TensorflowBackendV1.handle_batch_normalization(node, input_dict)

  @classmethod
  def handle_ceil(cls, node, input_dict):
    return TensorflowBackendV1.handle_trivial(node, input_dict)

  @classmethod
  def handle_clip(cls, node, input_dict):
    return TensorflowBackendV1.handle_clip(node, input_dict)

  @classmethod
  def handle_div(cls, node, input_dict):
    return TensorflowBackendV1.handle_div(node, input_dict)

  @classmethod
  def handle_dropout(cls, node, input_dict):
    return TensorflowBackendV1.handle_dropout(node, input_dict)

  @classmethod
  def handle_elu(cls, node, input_dict):
    return TensorflowBackendV1.handle_elu(node, input_dict)

  @classmethod
  def handle_exp(cls, node, input_dict):
    return TensorflowBackendV1.handle_trivial(node, input_dict)

  @classmethod
  def handle_floor(cls, node, input_dict):
    return TensorflowBackendV1.handle_trivial(node, input_dict)

  @classmethod
  def handle_gemm(cls, node, input_dict):
    return TensorflowBackendV1.handle_gemm(node, input_dict)

  @classmethod
  def handle_hard_sigmoid(cls, node, input_dict):
    return TensorflowBackendV1.handle_hard_sigmoid(node, input_dict)

  @classmethod
  def handle_leaky_relu(cls, node, input_dict):
    return TensorflowBackendV1.handle_leaky_relu(node, input_dict)

  @classmethod
  def handle_log(cls, node, input_dict):
    return TensorflowBackendV1.handle_trivial(node, input_dict)

  @classmethod
  def handle_max(cls, node, input_dict):
    return TensorflowBackendV1.handle_max(node, input_dict)

  @classmethod
  def handle_mean(cls, node, input_dict):
    return TensorflowBackendV1.handle_mean(node, input_dict)

  @classmethod
  def handle_min(cls, node, input_dict):
    return TensorflowBackendV1.handle_min(node, input_dict)

  @classmethod
  def handle_mul(cls, node, input_dict):
    return TensorflowBackendV1.handle_mul(node, input_dict)

  @classmethod
  def handle_neg(cls, node, input_dict):
    return TensorflowBackendV1.handle_trivial(node, input_dict)

  @classmethod
  def handle_p_relu(cls, node, input_dict):
    return TensorflowBackendV1.handle_p_relu(node, input_dict)

  @classmethod
  def handle_reciprocal(cls, node, input_dict):
    return TensorflowBackendV1.handle_trivial(node, input_dict)

  @classmethod
  def handle_relu(cls, node, input_dict):
    return TensorflowBackendV1.handle_trivial(node, input_dict)

  @classmethod
  def handle_selu(cls, node, input_dict):
    return TensorflowBackendV1.handle_selu(node, input_dict)

  @classmethod
  def handle_sigmoid(cls, node, input_dict):
    return TensorflowBackendV1.handle_trivial(node, input_dict)

  @classmethod
  def handle_sqrt(cls, node, input_dict):
    return TensorflowBackendV1.handle_trivial(node, input_dict)

  @classmethod
  def handle_sub(cls, node, input_dict):
    return TensorflowBackendV1.handle_sub(node, input_dict)

  @classmethod
  def handle_sum(cls, node, input_dict):
    return TensorflowBackendV1.handle_sum(node, input_dict)

  @classmethod
  def handle_tanh(cls, node, input_dict):
    return TensorflowBackendV1.handle_trivial(node, input_dict)