"""Frontend for exporting Tensorflow graph to ONNX graph

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from onnx_tf.frontend import TensorflowFrontendBase
from onnx import helper

register_onnx_op = TensorflowFrontendBase.register_onnx_op


class TensorflowFrontend(TensorflowFrontendBase):
  """ Tensorflow Frontend for ONNX
  """

  @classmethod
  @register_onnx_op("Pad")
  def handle_pad(cls, node, **kwargs):
    consts = kwargs["consts"]
    assert node.inputs[1] in consts.keys()
    supported_modes = ["constant", "reflect"]
    mode = node.attr.get("mode", "constant")
    assert mode.lower() in supported_modes
    pads = np.transpose(consts[node.inputs[1]]).flatten()

    return helper.make_node(
        "Pad", [node.inputs[0]], [node.name],
        name=node.name,
        pads=pads,
        mode=mode,
        value=0.0)

  @classmethod
  @register_onnx_op("Split")
  def handle_split_v(cls, node, **kwargs):
    consts = kwargs["consts"]
    split = consts[node.inputs[1]]
    axis = int(consts[node.inputs[2]])
    output_names = [
        node.name + ":{}".format(i) if i > 0 else node.name
        for i in range(len(split))
    ]
    return helper.make_node(
        "Split", [node.inputs[0]], output_names, split=split, axis=axis)
