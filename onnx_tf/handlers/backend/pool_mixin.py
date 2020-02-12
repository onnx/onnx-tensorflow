import warnings

import tensorflow as tf

from onnx_tf.common import exception
from onnx_tf.common import get_data_format
from onnx_tf.common import get_perm_from_formats
from .dilated_pooling import DilatedPooling
from onnx_tf.common.pooling_helper import py_pool


class PoolMixin(object):

  @classmethod
  def pool(cls, node, input_dict, pooling_type, strict=True):
    x = input_dict[node.inputs[0]]
    orig_x = x

    kernel_shape = node.attrs["kernel_shape"]

    spatial_size = len(kernel_shape)
    x_rank = spatial_size + 2

    kernel_shape = node.attrs["kernel_shape"]
    strides = node.attrs.get("strides", [1] * spatial_size)
    dilations = node.attrs.get("dilations", [1] * spatial_size)
    ceil_mode = bool(node.attrs.get("ceil_mode", 0))
    pads = node.attrs.get("auto_pad", "NOTSET")
    if pads == "NOTSET":
      pads = node.attrs.get("pads", [0] * spatial_size * 2)

    count_include_pad = bool(node.attrs.get("count_include_pad", 0))
    if pooling_type == "AVG":
      pooling_name = "AveragePool"
    elif pooling_type == "MAX":
      pooling_name = "MaxPool"
    elif pooling_type == "MAX_WITH_ARGMAX":
      pooling_name = "MaxPoolWithArgmax"

    if spatial_size > 3:
      exception.OP_UNSUPPORTED_EXCEPT(
          pooling_name + " with {}D input".format(x_rank), "Tensorflow")
    if pooling_type == "MAX_WITH_ARGMAX" and x_rank != 4:
      exception.OP_UNSUPPORTED_EXCEPT(
          pooling_name + " with {}D input".format(x_rank), "Tensorflow")
    if node.attrs.get("storage_order", 0) != 0:
      exception.OP_UNSUPPORTED_EXCEPT(pooling_name + " with column major",
                                      "Tensorflow")

    storage_format, _ = get_data_format(x_rank)

    need_trans = storage_format.startswith("NC")
    if need_trans:
      compute_format = "N" + storage_format[2:] + "C"
      x = tf.transpose(
          x, perm=get_perm_from_formats(storage_format, compute_format))

    dp = DilatedPooling(
        input=x,
        kernel_shape=kernel_shape,
        strides=strides,
        dilations=dilations,
        padding=pads,
        ceil_mode=ceil_mode,
        pooling_type=pooling_type,
        count_include_pad=count_include_pad)
    if not dp.is_supported():
      if strict:
        warnings.warn("Using the pooling op in compatibility mode. "
                      "This means your graph cannot be serialized.",
                      UserWarning)

        return [
            tf.compat.v1.py_func(py_pool, [
                orig_x, kernel_shape, strides, dilations, pads, ceil_mode,
                "AVG", False
            ], orig_x.dtype)
        ]
      else:
        exception.OP_UNSUPPORTED_EXCEPT("strict == 0 and average pool"
                                        " arguments not compatible",
                                        "Tensorflow")

    def dilated_pool():
      return (dp.dilated_pool(), None)

    # select correct op depending on the pooling type
    pooling_op = dilated_pool if pooling_type in ["MAX", "AVG"] else \
        dp.dilated_maxpool_with_argmax

    # select the correct transpose ops depending on the input storage format
    perm = get_perm_from_formats(compute_format, storage_format)

    def postprocess(pooled, argmax):
      return (tf.transpose(pooled, perm=perm) if need_trans else pooled,
              tf.transpose(argmax, perm=perm)
              if need_trans and argmax is not None else argmax)

    pooled, argmax = pooling_op()
    pooled, argmax = postprocess(pooled, argmax)

    result = [pooled] if argmax is None else [pooled, argmax]

    return result
