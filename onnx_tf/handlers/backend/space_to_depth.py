import copy

import tensorflow as tf

from onnx_tf.common import get_data_format
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("SpaceToDepth")
@tf_func(tf.nn.space_to_depth)
class SpaceToDepth(BackendHandler):

  @classmethod
  def get_attrs_processor_param(cls):
    return {"rename": {"blocksize": "block_size"}}

  @classmethod
  def version_1(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    x_rank = len(x.get_shape())
    storage_format, compute_format = get_data_format(x_rank)
    attrs = copy.deepcopy(node.attrs)
    attrs["data_format"] = storage_format
    return [
        cls.make_tensor_from_onnx_node(
            node, attrs=attrs, c_first_cuda_only=True, **kwargs)
    ]
