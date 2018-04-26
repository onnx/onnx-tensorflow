"""Frontend for exporting Tensorflow graph to ONNX graph

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx_tf.frontend import TensorflowFrontendBase
from onnx import helper, TensorProto
from onnx.onnx_pb2 import GraphProto, TensorProto, AttributeProto
import numpy as np

register_onnx_op = TensorflowFrontendBase.register_onnx_op

class TensorflowFrontend(TensorflowFrontendBase):
  """ Tensorflow Frontend for ONNX
  """

  @classmethod
  @register_onnx_op("Gather")
  def handle_gather(cls, node, **kwargs):
    return helper.make_node(
        "Gather",
        node.inputs, [node.name],
	name = node.name,
	axis = node.attr.get("axis", 0))

  @classmethod
  @register_onnx_op("GreaterEqual")
  def handle_greater_equal(cls, node, **kwargs):
    first_cond = helper.make_node(
        "Greater", inputs=node.inputs, outputs=[node.name + "_f"], broadcast=1)
    sec_cond = helper.make_node(
        "Equal", inputs=node.inputs, outputs=[node.name + "_s"], broadcast=1)
    final_cond = helper.make_node(
        "Or", inputs=[node.name + "_f", node.name + "_s"], outputs=[node.name], broadcast=1)
    return [first_cond, sec_cond, final_cond]

  @classmethod
  @register_onnx_op("Range")
  def handle_range(cls, node, **kwargs):
    consts = kwargs["consts"]
    assert node.inputs[0] in consts.keys()
    assert node.inputs[1] in consts.keys()
    assert node.inputs[2] in consts.keys()
    start = consts[node.inputs[0]]
    end = consts[node.inputs[1]]
    delta = consts[node.inputs[2]]
    ranged_data = np.array(range(start, end, delta))
    return helper.make_node(
        "Constant", inputs=[], outputs=[node.name], value=helper.make_tensor(
        name='const_tensor',
        data_type=TensorProto.FLOAT,
        dims=ranged_data.shape,
        vals=ranged_data
    ))
