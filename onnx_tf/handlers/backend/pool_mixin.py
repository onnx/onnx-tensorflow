import itertools
import warnings

try:
  from itertools import izip as zip
except ImportError:  # will be 3.x series
  pass

import numpy as np
import tensorflow as tf

from onnx_tf.common import exception
from onnx_tf.common import get_data_format
from onnx_tf.common import get_perm_from_formats
from onnx_tf.common import supports_device
from .pad_mixin import PadMixin

# Constant string used to indicate that requested padding
# is not natively supported in Tensorflow.
PAD_TF_INCOMPATIBLE = "PAD_TF_INCOMPATIBLE"


class PoolMixin(object):

  @classmethod
  def pool(cls, node, input_dict, pool_func, pooling_type, strict=True):
    x = input_dict[node.inputs[0]]
    x_rank = len(x.get_shape())
    x_shape = x.get_shape().as_list()
    spatial_size = x_rank - 2

    support_cuda = supports_device("CUDA")
    storage_format, compute_format = get_data_format(x_rank)

    kernel_shape = node.attrs["kernel_shape"]
    strides = node.attrs.get("strides", [1] * spatial_size)
    pads = node.attrs.get("pads", None)
    pad = PAD_TF_INCOMPATIBLE
    # from version 7
    count_include_pad = node.attrs.get("count_include_pad", 0)

    # If padding is specified, try to recover it from explicit padding
    # specification to tensorflow padding mode:
    if pads is not None:
      pad = cls._get_tf_pad(x_shape[2:], kernel_shape, strides, pads)
    else:
      # Neither pad nor auto_pad is specified, assume no padding.
      if "auto_pad" not in node.attrs:
        pad = "VALID"
      # We consult auto_pad if pad is not specified and auto_pad
      # is available.
      else:
        if node.attrs["auto_pad"] == "SAME_UPPER":
          pad = "SAME"
        elif node.attrs["auto_pad"] == "VALID":
          pad = "VALID"
        elif node.attrs["auto_pad"] == "SAME_LOWER":
          pad = PAD_TF_INCOMPATIBLE
        if count_include_pad == 1:
          _, pads = cls._pool_get_shapes(node.attrs["auto_pad"], x_shape[2:],
                                         kernel_shape, strides,
                                         [0] * spatial_size * 2)

    if pooling_type in ("AVG", "MAX"):
      if strict and count_include_pad == 0:
        if pad is PAD_TF_INCOMPATIBLE:
          return cls._compatibility_pool(node, input_dict, pooling_type)
      else:
        if pads != [0] * spatial_size * 2:
          x = PadMixin.get_padding_as_op(x, pads)
        pad = "VALID"
    elif pooling_type == "MAX_WITH_ARGMAX":
      if pad is PAD_TF_INCOMPATIBLE:
        exception.OP_UNSUPPORTED_EXCEPT(
            "MaxPoolWithArgmax with pad is None or incompatible mode",
            "Tensorflow")
      if x_rank != 4:
        exception.OP_UNSUPPORTED_EXCEPT(
            "MaxPoolWithArgmax with {}D input".format(x_rank), "Tensorflow")
      if node.attrs.get("storage_order", 0) != 0:
        exception.OP_UNSUPPORTED_EXCEPT("MaxPoolWithArgmax with column major",
                                        "Tensorflow")

      need_trans = storage_format != "NHWC"
      if need_trans:
        x = tf.transpose(x, perm=get_perm_from_formats(storage_format, "NHWC"))
      pooled, argmax = pool_func(
          x, [1] + kernel_shape + [1], padding=pad, strides=[1] + strides + [1])
      if need_trans:
        pooled = tf.transpose(
            pooled, perm=get_perm_from_formats("NHWC", storage_format))
        argmax = tf.transpose(
            argmax, perm=get_perm_from_formats("NHWC", storage_format))

      return [pooled, argmax]

    if support_cuda:
      pooled = pool_func(
          x,
          kernel_shape,
          padding=pad,
          strides=strides,
          data_format=compute_format)
    else:
      x = tf.transpose(
          x, perm=get_perm_from_formats(storage_format, compute_format))
      pooled = pool_func(
          x,
          kernel_shape,
          padding=pad,
          strides=strides,
          data_format=compute_format)
      pooled = tf.transpose(
          pooled, perm=get_perm_from_formats(compute_format, storage_format))

    return [pooled]

  @classmethod
  def _compatibility_pool(cls, node, input_dict, pooling_type):
    warnings.warn(
        "Using the pooling op in compatibility mode."
        "This means your graph cannot be serialized."
        "Please configure your pooling operation to only use paddings that "
        "correspond to Tensorflow SAME or VALID padding.", UserWarning)

    def py_pool(x, kernel_shape, strides, pads, out_shape, count_include_pad,
                pooling_type):
      pooling_type = pooling_type.decode('UTF-8')
      x_shape = np.shape(x)
      spatial_size = len(x_shape[2:])
      pad_attr = [(0, 0), (0, 0)] + [
          (pads[i], pads[i + spatial_size]) for i in range(spatial_size)
      ]
      constant_values = np.nan if count_include_pad == 0 else 0
      padded = np.pad(
          x, pad_attr, mode="constant", constant_values=constant_values)
      pad_shape = [
          pads[i] + pads[i + spatial_size] for i in range(spatial_size)
      ]

      y = np.zeros([x_shape[0], x_shape[1]] + list(out_shape))

      for shape in itertools.product(
          range(x_shape[0]), range(x_shape[1]), *[
              range(
                  int((x_shape[i + 2] + pad_shape[i] - kernel_shape[i]) /
                      strides[i] + 1)) for i in range(spatial_size)
          ]):
        window = padded[shape[0], shape[1]]
        window_vals = np.array([
            window[i] for i in list(
                itertools.product(*[
                    range(strides[i] * shape[i + 2], strides[i] * shape[i + 2] +
                          kernel_shape[i]) for i in range(spatial_size)
                ]))
        ])
        if pooling_type == 'AVG':
          f = np.average
        elif pooling_type == 'MAX':
          f = np.max
        else:
          raise NotImplementedError(
              'Pooling type {} does not support. Should be AVG, MAX'.format(
                  pooling_type))

        if count_include_pad == 0:
          y[shape] = f(window_vals[np.where(~np.isnan(window_vals))])
        else:
          y[shape] = f(window_vals)
      return y.astype(np.float32)

    x = input_dict[node.inputs[0]]
    x_shape = x.shape.as_list()
    spatial_size = len(x_shape) - 2
    kernel_shape = node.attrs["kernel_shape"]
    strides = node.attrs.get("strides", [1] * spatial_size)
    pads = node.attrs.get("pads", [0] * spatial_size * 2)
    auto_pad = node.attrs.get("auto_pad", "")
    count_include_pad = node.attrs.get("count_include_pad", 0)

    out_shape, pads = cls._pool_get_shapes(auto_pad, x_shape[2:], kernel_shape,
                                           strides, pads)

    pooled = tf.py_func(py_pool, [
        x, kernel_shape, strides, pads, out_shape, count_include_pad,
        pooling_type
    ], tf.float32)
    pooled.set_shape(x_shape[0:2] + out_shape)
    return [pooled]

  @classmethod
  def _pool_get_shapes(cls, auto_pad, x_shape, kernel_shape, strides, pads):

    def _get_pad_shape(auto_pad, input_spatial_shape, kernel_spatial_shape,
                       strides_spatial, output_spatial_shape):
      pad_shape = [0] * len(input_spatial_shape)
      if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        for i in range(len(input_spatial_shape)):
          pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial[i] + kernel_spatial_shape[i] - \
                         input_spatial_shape[i]
      elif auto_pad in ("VALID", ""):
        pass
      return pad_shape

    def _get_output_shape(auto_pad, input_spatial_shape, kernel_spatial_shape,
                          strides_spatial):
      out_shape = [0] * len(input_spatial_shape)
      if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        for i in range(len(input_spatial_shape)):
          out_shape[i] = int(
              np.ceil(
                  float(input_spatial_shape[i]) / float(strides_spatial[i])))
      elif auto_pad in ("VALID", ""):
        for i in range(len(input_spatial_shape)):
          out_shape[i] = int(
              np.ceil(
                  float(input_spatial_shape[i] - (kernel_spatial_shape[i] - 1))
                  / float(strides_spatial[i])))
      return out_shape

    spatial_size = len(x_shape)
    new_pads = pads[:]
    if auto_pad in ["SAME_UPPER", "SAME_LOWER"]:
      out_shape = _get_output_shape(auto_pad, x_shape, kernel_shape, strides)
      pad_shape = _get_pad_shape(auto_pad, x_shape, kernel_shape, strides,
                                 out_shape)
      for i in range(spatial_size):
        if auto_pad == "SAME_LOWER":
          new_pads[i + spatial_size] = pad_shape[i] // 2
          new_pads[i] = pad_shape[i] - new_pads[i + spatial_size]
        elif auto_pad == "SAME_UPPER":
          new_pads[i] = pad_shape[i] // 2
          new_pads[i + spatial_size] = pad_shape[i] - new_pads[i]
    elif auto_pad in ["", "VALID"]:
      pad_shape = [
          pads[i] + pads[i + spatial_size] for i in range(spatial_size)
      ]
      out_shape = _get_output_shape(auto_pad, np.add(x_shape, pad_shape),
                                    kernel_shape, strides)
    return out_shape, new_pads

  # input_shape, kernel_shape, strides are specified for
  # spatial dims only.
  @classmethod
  def _get_tf_pad(cls, input_shape, kernel_shape, strides, pads):
    assert pads is not None
    num_sp_dim = int(len(kernel_shape))

    if pads == [0] * num_sp_dim * 2:
      return "VALID"

    _, same_pads = cls._pool_get_shapes("SAME_UPPER", input_shape, kernel_shape,
                                        strides, pads)
    if pads == same_pads:
      return "SAME"

    return PAD_TF_INCOMPATIBLE
