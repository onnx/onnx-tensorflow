import tensorflow as tf

from onnx_tf.common import exception
from onnx_tf.common import get_perm_from_formats
from onnx_tf.common import logger
from onnx_tf.common import sys_config
from onnx_tf.common.pooling_helper import py_pool
from onnx_tf.common.pooling_helper import calc_pads_same
from onnx_tf.common.pooling_helper import calc_output_shape
from onnx_tf.common.tf_helper import tf_shape
from .dilated_pooling import DilatedPooling


class PoolMixin(object):

  @classmethod
  @tf.autograph.experimental.do_not_convert()
  def pool(cls, node, input_dict, pooling_type, strict=True):
    x = input_dict[node.inputs[0]]

    kernel_shape = node.attrs["kernel_shape"]

    spatial_size = len(kernel_shape)
    x_rank = spatial_size + 2

    kernel_shape = node.attrs["kernel_shape"]
    strides = node.attrs.get("strides", [1] * spatial_size)
    dilations = node.attrs.get("dilations", [1] * spatial_size)
    ceil_mode = bool(node.attrs.get("ceil_mode", 0))
    pads = node.attrs.get("auto_pad", "NOTSET")
    p = node.attrs.get("p", 2)

    if pads == "NOTSET":
      pads = node.attrs.get("pads", [0] * spatial_size * 2)
      # In case shape is fully defined, check if pads match
      # SAME padding in Tensorflow
      if x.shape.is_fully_defined() and pads != [0] * spatial_size * 2:
        in_shape = x.get_shape()
        same_paddings = calc_pads_same(in_shape[1:x_rank - 1], kernel_shape,
                                       strides, dilations, "SAME_UPPER")
        if pads == same_paddings:
          pads = "SAME_UPPER"

    count_include_pad = bool(node.attrs.get("count_include_pad", 0))
    if pooling_type == "AVG":
      pooling_name = "AveragePool"
    elif pooling_type == "MAX":
      pooling_name = "MaxPool"
    elif pooling_type == "MAX_WITH_ARGMAX":
      pooling_name = "MaxPoolWithArgmax"
    elif pooling_type == "LP":
      pooling_name = "LpPool"

    if spatial_size > 3:
      exception.OP_UNSUPPORTED_EXCEPT(
          pooling_name + " with {}D input".format(x_rank), "Tensorflow")
    if pooling_type == "MAX_WITH_ARGMAX" and x_rank != 4:
      exception.OP_UNSUPPORTED_EXCEPT(
          pooling_name + " with {}D input".format(x_rank), "Tensorflow")
    if node.attrs.get("storage_order", 0) != 0:
      exception.OP_UNSUPPORTED_EXCEPT(pooling_name + " with column major",
                                      "Tensorflow")

    x_dtype = x.dtype
    # For max_pool and max_pool_with_argmax tensoflow don't support
    # NCHW data format input in int8 or uint8 datatype, therefore
    # need to cast to float16 in order to run with NCHW data format
    need_cast = pooling_type in [
        'MAX', 'MAX_WITH_ARGMAX'
    ] and sys_config.device == 'CUDA' and x_dtype in [tf.int8, tf.uint8]
    x = tf.cast(x, tf.float16) if need_cast else x

    dp = DilatedPooling(input=x,
                        kernel_shape=kernel_shape,
                        strides=strides,
                        dilations=dilations,
                        padding=pads,
                        ceil_mode=ceil_mode,
                        pooling_type=pooling_type,
                        count_include_pad=count_include_pad,
                        p=p)
    if not dp.is_supported():
      if strict:
        logger.warning("Using the pooling op in compatibility mode. "
                       "This means your graph cannot be serialized.")

        result = tf.numpy_function(py_pool, [
            x, kernel_shape, strides, dilations, pads, ceil_mode, pooling_type,
            False
        ], x.dtype)

        if x.shape.is_fully_defined():
          shape = x.get_shape()
          output_shape = shape[0:2] + calc_output_shape(
              shape[2:x_rank], kernel_shape, strides, dilations, pads,
              ceil_mode)
        else:
          output_shape = [None] * x_rank
        result.set_shape(output_shape)
        return [result]
      else:
        exception.OP_UNSUPPORTED_EXCEPT(
            "strict == 0 and " + pooling_name + " arguments not compatible",
            "Tensorflow")

    from absl import logging
    logging.set_verbosity(logging.INFO)

    def dilated_pool():
      return (dp.dilated_pool(), None)

    # select correct op depending on the pooling type
    pooling_op = dilated_pool if pooling_type in ["MAX", "AVG", "LP"] else \
        dp.dilated_maxpool_with_argmax

    def postprocess(pooled, argmax):

      def convert_NHWC_indices_to_NCHW_indices(argmax):
        # i - index in NCHW
        # I - index in NHWC
        # C - number of channels
        # b - batch = I // CHW
        # c - channel = I % C
        # H - height
        # W - weight
        # I = i - c(HW - 1) + (C - 1)(i - bCHW - cHW)
        # i = (I + c(HW - 1) + (C - 1)(bCHW + cHW))/C

        # x_shape will always be in NCHW format here,
        # because maxpool_with_argmax only support 2d input
        x_shape = tf_shape(x)
        N = x_shape[0]
        C = x_shape[1]
        H = x_shape[2]
        W = x_shape[3]
        HW = tf.math.multiply(H, W)
        CHW = tf.math.multiply(C, HW)
        argmax_b = tf.math.floordiv(argmax, CHW)
        argmax_c = tf.math.floormod(argmax, C)
        new_ind = tf.math.add(
            argmax, tf.math.multiply(argmax_c, tf.math.subtract(HW, 1)))
        new_ind = tf.math.add(
            new_ind,
            tf.math.multiply(
                tf.math.subtract(C, 1),
                tf.math.add(tf.math.multiply(argmax_b, CHW),
                            tf.math.multiply(argmax_c, HW))))
        new_ind = tf.math.floordiv(new_ind, C)

        # add batch dimension into the argmax index
        batch_offsets = tf.math.multiply(tf.range(N, dtype=new_ind.dtype), CHW)
        for _ in range(new_ind.shape.rank - 1):
          batch_offsets = tf.expand_dims(batch_offsets, -1)
        new_ind = tf.math.add(new_ind, batch_offsets)

        return new_ind

      if argmax is not None:
        argmax = convert_NHWC_indices_to_NCHW_indices(argmax)

      # select the correct transpose ops depending on the input storage format
      perm = get_perm_from_formats(dp.compute_format, dp.storage_format)

      pooled = tf.transpose(pooled, perm=perm) if dp.need_trans else pooled
      pooled = tf.cast(pooled, x_dtype) if need_cast else pooled
      argmax = tf.transpose(
          argmax, perm=perm) if dp.need_trans and argmax is not None else argmax

      return pooled, argmax

    pooled, argmax = pooling_op()
    pooled, argmax = postprocess(pooled, argmax)

    result = [pooled] if argmax is None else [pooled, argmax]

    return result
