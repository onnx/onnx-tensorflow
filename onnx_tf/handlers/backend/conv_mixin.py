import tensorflow as tf

from onnx_tf.common import get_data_format
from onnx_tf.common import get_perm_from_formats
from onnx_tf.common import supports_device
from .broadcast_mixin import BroadcastMixin
from .pad_mixin import PadMixin


class ConvMixin(BroadcastMixin):

  @classmethod
  def conv(cls, node, input_dict, transpose=False):
    """ Convolution method for both conv and transposed conv
    For transposed conv,
      Attr pads is not used for input, but declares how much output is padded.
      Here, output means output from transposed conv which already pad output_padding if set.
      So the pseudo explanation for output should be:
        output = conv_transpose_output + output_padding - pads
      And conv_transpose_output shape should be:
        conv_transpose_output_shape[i] = strides[i] * (input_shape[i] - 1) + kernel_shape[i]
    """
    x = input_dict[node.inputs[0]]
    x_rank = len(x.get_shape())
    x_shape = x.get_shape().as_list()
    spatial_size = x_rank - 2

    support_cuda = supports_device("CUDA")
    storage_format, compute_format = get_data_format(x_rank)
    compute_c_idx = compute_format.find("C")
    spatial_format = "".join([d for d in compute_format if d not in ["N", "C"]])

    in_weights = input_dict[node.inputs[1]]
    weights_rank = len(in_weights.get_shape())
    if transpose:
      # Translate weights from (C x M x KH x KW) to (KH x KW X M X C)
      perm = list(range(2, weights_rank)) + [1, 0]
    else:
      # Translate weights from (M x C x KH x KW) to (KH x KW X C X M)
      perm = list(range(2, weights_rank)) + [1, 0]

    if "kernel_shape" in node.attrs.keys():
      kernel_shape = node.attrs["kernel_shape"]
      assert in_weights.get_shape().as_list()[2:] == kernel_shape, (
          "kernel_shape "
          "attr of convolution does not match the actual weight "
          "passed to this operation, attr {}, actual {}").format(
              kernel_shape,
              in_weights.get_shape().as_list())

    weights = tf.transpose(in_weights, perm)
    dilations = node.attrs.get("dilations", [1] * spatial_size)
    strides = node.attrs.get("strides", [1] * spatial_size)

    pads = node.attrs.get("pads", [0, 0] * spatial_size)

    if not transpose:
      x = PadMixin.get_padding_as_op(x, pads)

    group = node.attrs.get("group", 1)

    weight_groups = tf.split(weights, num_or_size_splits=group, axis=-1)

    if support_cuda:
      xs = tf.split(x, num_or_size_splits=group, axis=1)
    else:
      x = tf.transpose(
          x, perm=get_perm_from_formats(storage_format, compute_format))
      xs = tf.split(x, num_or_size_splits=group, axis=-1)

    if transpose:
      if dilations != [1] * spatial_size:
        raise RuntimeError("Cannot set non-1 dilation for conv transpose.")
      convolved = []
      for (x, weight) in zip(xs, weight_groups):
        x_spatial_shape = [
            x_shape[storage_format.find(d)] for d in spatial_format
        ]
        weights_shape = weights.get_shape().as_list()

        # calculate output shape
        output_shape = node.attrs.get("output_shape", None)
        conv_output_shape = [x_shape[storage_format.find("N")]]
        if output_shape is None:
          conv_output_shape += [
              strides[i] * (x_spatial_shape[i] - 1) + weights_shape[i]
              for i in list(range(spatial_size))
          ]
        else:
          conv_output_shape += [
              s + pads[i] + pads[spatial_size + i]
              for i, s in enumerate(output_shape[-2:])
          ]
        conv_output_shape.insert(compute_c_idx, weights_shape[-2])

        # make strides to match input rank
        strides_full = [1] + strides
        strides_full.insert(compute_c_idx, 1)

        # get corresponding function in tf
        if spatial_size == 1:
          conv_func = tf.contrib.nn.conv1d_transpose
          strides_full = strides[0]
        elif spatial_size == 2:
          conv_func = tf.nn.conv2d_transpose
        elif spatial_size == 3:
          conv_func = tf.nn.conv3d_transpose
        else:
          raise NotImplementedError(
              "Transposed convolution for {}d is not implemented in Tensorflow".
              format(spatial_size))

        # use raw input x to do transposed conv
        conv_rs = conv_func(
            x,
            weights,
            conv_output_shape,
            strides_full,
            padding="VALID",
            data_format=compute_format)

        # pad output first by output_padding attr
        if "output_padding" in node.attrs and output_shape is None:
          output_padding = [[0, 0]
                           ] + [[0, p] for p in node.attrs["output_padding"]]
          output_padding.insert(compute_c_idx, [0, 0])
          conv_rs = tf.pad(conv_rs, output_padding)

        # remove pads set in pads attr
        conv_rs_shape = conv_rs.get_shape().as_list()
        begin = [0] + pads[:spatial_size]
        begin.insert(compute_c_idx, 0)
        size = [
            s if d in ["N", "C"] else s - pads[spatial_format.find(d)] -
            pads[spatial_format.find(d) + spatial_size]
            for d, s in zip(compute_format, conv_rs_shape)
        ]
        conv_rs = tf.slice(conv_rs, begin=begin, size=size)

        convolved.append(conv_rs)
    else:
      convolved = [
          tf.nn.convolution(
              x,
              weight,
              "VALID",
              strides=strides,
              dilation_rate=dilations,
              data_format=compute_format)
          for (x, weight) in zip(xs, weight_groups)
      ]

    if len(node.inputs) == 2:
      if support_cuda:
        output = tf.concat(convolved, axis=1)
      else:
        output = tf.concat(convolved, axis=-1)
        output = tf.transpose(
            output, perm=get_perm_from_formats(compute_format, storage_format))
    else:
      bias = input_dict[node.inputs[2]]
      bias = cls.explicit_broadcast([x, bias], compute_c_idx)

      if support_cuda:
        output = tf.concat(convolved, axis=1)
        output = tf.add(output, bias)
      else:
        output = tf.concat(convolved, axis=-1)
        output = tf.add(output, bias)
        output = tf.transpose(
            output, perm=get_perm_from_formats(compute_format, storage_format))

    return [output]
