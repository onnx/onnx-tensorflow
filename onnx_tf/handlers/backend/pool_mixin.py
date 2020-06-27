import tensorflow as tf

from onnx_tf.common import exception
from onnx_tf.common import get_data_format
from onnx_tf.common import get_perm_from_formats
from onnx_tf.common import logger 
from .dilated_pooling import DilatedPooling
from onnx_tf.common.pooling_helper import py_pool
from onnx_tf.common.pooling_helper import calc_pads_same
from onnx_tf.common.pooling_helper import calc_output_shape

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
    p = node.attrs.get("p", 2)

    if pads == "NOTSET":
      pads = node.attrs.get("pads", [0] * spatial_size * 2)
      # In case shape is fully defined, check if pads match
      # SAME padding in Tensorflow
      if x.shape.is_fully_defined() and pads != [0] * spatial_size * 2:
        in_shape = x.get_shape()
        same_paddings = calc_pads_same(in_shape[1:x_rank-1], kernel_shape,
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
        count_include_pad=count_include_pad,
        p=p)
    if not dp.is_supported():
      if strict:
        logger.warning("Using the pooling op in compatibility mode. "
                      "This means your graph cannot be serialized.")

        result = tf.numpy_function(py_pool, [
                orig_x, kernel_shape, strides, dilations, pads, ceil_mode,
                pooling_type, False
            ], orig_x.dtype)

        if orig_x.shape.is_fully_defined():
          shape = orig_x.get_shape()
          output_shape = shape[0:2] + calc_output_shape(shape[2:x_rank],
                  kernel_shape, strides, dilations, pads, ceil_mode)
        else:
          output_shape = [None] * x_rank
        result.set_shape(output_shape)
        return [result]
      else:
        exception.OP_UNSUPPORTED_EXCEPT("strict == 0 and " + pooling_name +
                                        " arguments not compatible",
                                        "Tensorflow")

    def dilated_pool():
      return (dp.dilated_pool(), None)

    # select correct op depending on the pooling type
    pooling_op = dilated_pool if pooling_type in ["MAX", "AVG", "LP"] else \
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
