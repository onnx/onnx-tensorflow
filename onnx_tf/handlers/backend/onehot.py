import copy
import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("OneHot")
@tf_func(tf.one_hot)
class OneHot(BackendHandler):

  @classmethod
  def version_9(cls, node, **kwargs):
    attrs = copy.deepcopy(node.attrs)
    tensor_dict = kwargs["tensor_dict"]
    indices = tensor_dict[node.inputs[0]]
    depth = tensor_dict[node.inputs[1]][0]
    off_value = tensor_dict[node.inputs[2]][0]
    on_value = tensor_dict[node.inputs[2]][1]
    attrs["dtype"] = on_value.dtype
    return [
        cls.make_tensor_from_onnx_node(
            node,
            inputs=[indices, depth, on_value, off_value],
            attrs=attrs,
            **kwargs)
    ]
