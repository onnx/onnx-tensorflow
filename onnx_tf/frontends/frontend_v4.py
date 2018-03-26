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

  ONNX_TO_HANDLER = {
      "concat": "concat_v2",
  }

  @classmethod
  def handle_concat_v2(cls, node, **kwargs):
    consts = kwargs["consts"]
    assert node.inputs[-1] in consts.keys()
    axis = int(consts[node.inputs[-1]])
    return helper.make_node(
        "Concat", inputs=node.inputs[0:-1], outputs=[node.name], axis=axis)
