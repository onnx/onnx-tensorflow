import numpy as np

from onnx.helper import make_node
from onnx.numpy_helper import from_array

from onnx_tf.common import exception
from onnx_tf.common import get_unique_suffix
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("OneHot")
@tf_op("OneHot")
class OneHot(FrontendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    if node.inputs[2] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[2], node.op_type)
    if node.inputs[3] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[3], node.op_type)

  @classmethod
  def version_9(cls, node, **kwargs):
    indices = node.inputs[0]
    depth = node.inputs[1]
    axis = node.attr.get('axis', -1)

    import pdb; pdb.set_trace()
    on_value = kwargs['consts'][node.inputs[2]].item(0)
    off_value = kwargs['consts'][node.inputs[3]].item(0)
    values = np.array([off_value, on_value])
    constant_output_name = node.outputs[0] + '_' + get_unique_suffix()

    constant_node = make_node(
        'Constant',
        inputs=[],
        outputs=[constant_output_name],
        value=from_array(values))

    onehot_node = cls.make_node_from_tf_node(
        node, [indices, depth, constant_output_name], axis=axis)

    return [constant_node, onehot_node]
