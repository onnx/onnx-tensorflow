import copy

import tensorflow as tf

from onnx_tf.common import get_data_format
from onnx_tf.common.tf_helper import tf_shape
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("DepthToSpace")
@tf_func(tf.nn.depth_to_space)
class DepthToSpace(BackendHandler):

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
        cls.make_tensor_from_onnx_node(node,
                                       attrs=attrs,
                                       c_first_cuda_only=True,
                                       **kwargs)
    ]

  @classmethod
  def _common(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    x_rank = len(x.get_shape())
    storage_format, _ = get_data_format(x_rank)
    attrs = copy.deepcopy(node.attrs)
    attrs["data_format"] = storage_format
    mode = attrs.get("mode", "DCR")

    if mode == "CRD":
      # need native computation
      bsize = attrs.get("blocksize")
      x_shape = tf_shape(x)
      batch, channel = x_shape[0], x_shape[1]
      height, width = x_shape[2], x_shape[3]
      csize = channel // (bsize**2)

      reshape_node = tf.reshape(x, [batch, csize, bsize, bsize, height, width])
      transpose_node = tf.transpose(reshape_node, perm=[0, 1, 4, 2, 5, 3])
      return [
          tf.reshape(transpose_node,
                     [batch, csize, height * bsize, width * bsize])
      ]

    return [
        cls.make_tensor_from_onnx_node(node,
                                       attrs=attrs,
                                       c_first_cuda_only=True,
                                       **kwargs)
    ]

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
