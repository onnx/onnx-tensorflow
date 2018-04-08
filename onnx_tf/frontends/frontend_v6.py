"""Frontend for exporting Tensorflow graph to ONNX graph

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx_tf.frontend import TensorflowFrontendBase
from onnx import helper


class TensorflowFrontend(TensorflowFrontendBase):
  """ Tensorflow Frontend for ONNX
  """

  @classmethod
  @TensorflowFrontendBase.register_onnx_op("batch_normalization")
  def handle_fused_batch_norm(cls, node, **kwargs):
    return helper.make_node(
        "BatchNormalization",
        node.inputs, [node.name],
        epsilon=node.attr.get("epsilon", 1e-5),
        is_test=node.attr.get("is_training", 0))
