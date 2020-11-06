import copy
import tensorflow as tf

from onnx_tf.common import exception
from onnx_tf.common import data_type
from onnx_tf.common import sys_config
from onnx_tf.common.tf_helper import tf_shape
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("OneHot")
@tf_func(tf.one_hot)
class OneHot(BackendHandler):
  depth_supported_type = [tf.int32]
  depth_cast_map = {
      tf.uint8: tf.int32,
      tf.uint16: tf.int32,
      tf.int8: tf.int32,
      tf.int16: tf.int32
  }

  @classmethod
  def args_check(cls, node, **kwargs):
    # update cast_map base on auto_cast flag
    cls.depth_cast_map[tf.uint32] = tf.int32 if sys_config.auto_cast else None
    cls.depth_cast_map[tf.uint64] = tf.int32 if sys_config.auto_cast else None
    cls.depth_cast_map[tf.int64] = tf.int32 if sys_config.auto_cast else None
    cls.depth_cast_map[tf.float16] = tf.int32 if sys_config.auto_cast else None
    cls.depth_cast_map[tf.float32] = tf.int32 if sys_config.auto_cast else None
    cls.depth_cast_map[tf.float64] = tf.int32 if sys_config.auto_cast else None

    tensor_dict = kwargs["tensor_dict"]
    depth = tensor_dict[node.inputs[1]]
    depth_dtype = depth.dtype
    if depth_dtype in cls.depth_cast_map and cls.depth_cast_map[
        depth_dtype] is None:
      exception.DTYPE_NOT_CAST_EXCEPT(
          "OneHot input " + node.inputs[1] + " with data type '" +
          data_type.tf_to_np_str(depth_dtype) + "'",
          data_type.tf_to_np_str_list(cls.depth_supported_type))

  @classmethod
  def process_neg_indices(cls, depth, indices):
    indices_dtype = indices.dtype
    depth_dtype = depth.dtype
    indices = tf.math.floormod(tf.add(tf.cast(indices, depth_dtype), depth),
                               depth)
    return tf.cast(indices, indices_dtype)

  @classmethod
  def _common(cls, node, **kwargs):
    attrs = copy.deepcopy(node.attrs)
    tensor_dict = kwargs["tensor_dict"]
    indices = tensor_dict[node.inputs[0]]
    depth = tensor_dict[node.inputs[1]]
    axis = attrs.get("axis", -1)

    # poocess negative axis
    axis = axis if axis >= 0 else len(tf_shape(indices)) + axis + 1

    # cast indices to tf.int64 according to the onnx spec
    indices = tf.cast(indices, tf.int64) if indices.dtype not in [
        tf.uint8, tf.int32, tf.int64
    ] else indices

    # process tf.one_hot unsupported datatype for depth
    depth = tf.cast(depth, cls.depth_cast_map[
        depth.dtype]) if depth.dtype in cls.depth_cast_map else depth

    # depth can be either a scalar or a 1D tensor of size 1 according
    # to ONNX schema, although operators doc states only scalar.
    # So we support both now.
    depth = tf.squeeze(depth) if len(tf_shape(depth)) == 1 else depth

    # process negative indices
    indices = cls.process_neg_indices(depth, indices)

    off_value = tensor_dict[node.inputs[2]][0]
    on_value = tensor_dict[node.inputs[2]][1]
    attrs["dtype"] = on_value.dtype
    attrs["axis"] = axis
    return [
        cls.make_tensor_from_onnx_node(
            node,
            inputs=[indices, depth, on_value, off_value],
            attrs=attrs,
            **kwargs)
    ]

  @classmethod
  def version_9(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)
