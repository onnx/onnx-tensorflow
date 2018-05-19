"""Frontend for exporting Tensorflow graph to ONNX graph

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx_tf.frontend import TensorflowFrontendBase
from onnx_tf.frontends.frontend_v1 import TensorflowFrontend as TensorflowFrontendV1
from onnx import helper

register_onnx_op = TensorflowFrontendBase.register_onnx_op


class TensorflowFrontend(TensorflowFrontendBase):
  """ Tensorflow Frontend for ONNX

  Lots of handler in this file just call frontend_v1 due to
  ONNX removed `consumed_inputs` attr for them which is not used in handler.
  """

  @classmethod
  @register_onnx_op("Add")
  def handle_add(cls, node, **kwargs):
    return TensorflowFrontendV1.handle_trivial(node, **kwargs)

  @classmethod
  @register_onnx_op("Add")
  def handle_bias_add(cls, node, **kwargs):
    return TensorflowFrontendV1.handle_bias_add(node, **kwargs)

  @classmethod
  @register_onnx_op("Ceil")
  def handle_ceil(cls, node, **kwargs):
    return TensorflowFrontendV1.handle_trivial(node, **kwargs)

  @classmethod
  @register_onnx_op("Exp")
  def handle_exp(cls, node, **kwargs):
    return TensorflowFrontendV1.handle_trivial(node, **kwargs)

  @classmethod
  @register_onnx_op("Floor")
  def handle_floor(cls, node, **kwargs):
    return TensorflowFrontendV1.handle_trivial(node, **kwargs)

  @classmethod
  @register_onnx_op("BatchNormalization")
  def handle_fused_batch_norm(cls, node, **kwargs):
    return helper.make_node(
        "BatchNormalization",
        node.inputs, [node.name],
        epsilon=node.attr.get("epsilon", 1e-5),
        is_test=node.attr.get("is_training", 0))

  @classmethod
  @register_onnx_op("Log")
  def handle_log(cls, node, **kwargs):
    return TensorflowFrontendV1.handle_trivial(node, **kwargs)

  @classmethod
  @register_onnx_op("And")
  def handle_logical_and(cls, node, **kwargs):
    return TensorflowFrontendV1.handle_trivial(node, **kwargs)

  @classmethod
  @register_onnx_op("Mul")
  def handle_mul(cls, node, **kwargs):
    return TensorflowFrontendV1.handle_trivial(node, **kwargs)

  @classmethod
  @register_onnx_op("Reciprocal")
  def handle_reciprocal(cls, node, **kwargs):
    return TensorflowFrontendV1.handle_trivial(node, **kwargs)

  @classmethod
  @register_onnx_op("Div")
  def handle_real_div(cls, node, **kwargs):
    return TensorflowFrontendV1.handle_trivial(node, **kwargs)

  @classmethod
  @register_onnx_op("Relu")
  def handle_relu(cls, node, **kwargs):
    return TensorflowFrontendV1.handle_trivial(node, **kwargs)

  @classmethod
  @register_onnx_op("Sigmoid")
  def handle_sigmoid(cls, node, **kwargs):
    return TensorflowFrontendV1.handle_trivial(node, **kwargs)

  @classmethod
  @register_onnx_op("Sqrt")
  def handle_sqrt(cls, node, **kwargs):
    return TensorflowFrontendV1.handle_trivial(node, **kwargs)

  @classmethod
  @register_onnx_op("Sub")
  def handle_sub(cls, node, **kwargs):
    return TensorflowFrontendV1.handle_trivial(node, **kwargs)

  @classmethod
  @register_onnx_op("Tanh")
  def handle_Tanh(cls, node, **kwargs):
    return TensorflowFrontendV1.handle_trivial(node, **kwargs)
