import copy

import tensorflow as tf

from onnx_tf.common import get_data_format
from onnx_tf.common import get_perm_from_formats
from onnx_tf.common import sys_config
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
  def _common(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    x_rank = len(x.get_shape())
    storage_format, compute_format = get_data_format(x_rank)
    attrs = copy.deepcopy(node.attrs)
    attrs["data_format"] = storage_format

    if sys_config.device == 'CUDA' and x.dtype not in {
        tf.uint8, tf.float16, tf.float32
    }:
      # Tensorflow GPU version doesn't support these datatype but CPU version support
      with tf.device("/cpu:0"):  # run it on cpu
        compute_format = compute_format.replace("C", "") + "C"
        pre_perm = get_perm_from_formats(storage_format, compute_format)
        post_perm = get_perm_from_formats(compute_format, storage_format)
        x_t = tf.transpose(x, perm=pre_perm)
        y = tf.nn.space_to_depth(x_t, attrs["blocksize"], compute_format)
        y = tf.transpose(y, perm=post_perm)
    else:
      y = cls.make_tensor_from_onnx_node(node,
                                         attrs=attrs,
                                         c_first_cuda_only=True,
                                         **kwargs)
    return [y]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
