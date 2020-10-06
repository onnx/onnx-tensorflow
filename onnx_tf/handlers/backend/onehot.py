import copy
import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.common.tf_helper import tf_shape


@onnx_op("OneHot")
@tf_func(tf.one_hot)
class OneHot(BackendHandler):

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

    # cast indices to tf.int64 and depth to tf.int32 if dtype is not
    # supported natively by Tensorflow. It is fairly safe since indices
    # and depth are integers
    indices = tf.cast(indices, tf.int64) if indices.dtype not in [
        tf.uint8, tf.int32, tf.int64
    ] else indices
    depth = tf.cast(depth, tf.int32) if depth.dtype not in [tf.int32] else depth

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
