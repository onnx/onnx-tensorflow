"""Frontend for exporting Tensorflow graph to ONNX graph

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx_tf.frontend import TensorflowFrontendBase
from onnx import helper

register_onnx_op = TensorflowFrontendBase.register_onnx_op


class TensorflowFrontend(TensorflowFrontendBase):
  """ Tensorflow Frontend for ONNX
  """

  @classmethod
  @register_onnx_op("Reshape")
  def handle_reshape(cls, node, **kwargs):
    return helper.make_node("Reshape", [node.inputs[0], node.inputs[1]],
                            [node.name])
