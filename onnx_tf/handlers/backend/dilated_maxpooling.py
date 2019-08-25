import tensorflow as tf
import math


def _calc_xy(v, k, d, s):
    return (v // k) * (s - k * d) + v * d


def _calc_orig_ind(ind, kH, kW, dH, dW, sH, sW, sizeW, in_width, pads):
    ind_shape = tf.shape(ind, out_type=tf.dtypes.int64)
    num_channels = ind_shape[3]

    # mod_floor op is not implemented on GPU
    # implement it using: a % b = a - (a // b) * b

    # nY = (ind // num_channels) // sizeW
    # nX = (ind // num_channels) % sizeW
    # ind_channel = ind % num_channels

    ind_ = ind // num_channels
    nY = ind_ // sizeW
    nX = ind_ - (ind_ // sizeW) * sizeW

    ind_channel = ind - ind_ * num_channels

    y = _calc_xy(nY, kH, dH, sH) - pads[0]
    x = _calc_xy(nX, kW, dW, sW) - pads[2]

    new_ind = num_channels * (y * in_width + x) + ind_channel
    return new_ind


def _calc_indexes(Y, X, nH, nW, sizeH, sizeW):
    Y = tf.reshape(Y * nW, [-1, 1])
    X = tf.reshape(X, [1, -1])

    ind = tf.reshape(Y + X, [sizeH * sizeW, 1])
    return ind


def _pad_input_same(input, strides, filterSizeH, filterSizeW,
                    padding="SAME_UPPER"):
    # Apply SAME padding to the input
    input_shape = tf.shape(input, out_type=tf.dtypes.int64)
    in_height = input_shape[1]
    in_width = input_shape[2]

    out_height = tf.cast(tf.math.ceil(in_height / strides[0]), tf.int64)
    out_width = tf.cast(tf.math.ceil(in_width / strides[1]), tf.int64)

    pad_along_height = tf.math.maximum((out_height - 1) * strides[0] +
                                       filterSizeH - in_height, 0)
    pad_along_width = tf.math.maximum((out_width - 1) * strides[1] +
                                      filterSizeW - in_width, 0)
    if padding.lower() == "same_lower":
        pad_op = tf.math.ceil
    else:
        pad_op = tf.math.floor
    pad_top = tf.cast(pad_op(pad_along_height / 2), dtype=tf.int64)
    pad_bottom = pad_along_height - pad_top
    pad_left = tf.cast(pad_op(pad_along_width / 2), dtype=tf.int64)
    pad_right = pad_along_width - pad_left

    tf_paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right],
                   [0, 0]]

    padded = tf.pad(input, tf_paddings, mode='CONSTANT',
                    constant_values=-math.inf)
    return (padded, [pad_top, pad_bottom, pad_left, pad_right])


def _pad_input_explicit(input, pads):
    if pads == [0] * 4:
        return (input, [0] * 4)
    tf_paddings = [[0, 0], [pads[0], pads[2]],
                   [pads[1], pads[3]], [0, 0]]
    padded = tf.pad(input, tf_paddings, mode='CONSTANT',
                    constant_values=-math.inf)
    return (padded, [pads[0], pads[2], pads[1], pads[3]])


def _calc_padding_ceil_mode(input, strides, filterSizeH, filterSizeW):
    input_shape = tf.shape(input, out_type=tf.dtypes.int64)
    in_height = input_shape[1]
    in_width = input_shape[2]

    out_h = (in_height - filterSizeH) / strides[0]
    pad_h = tf.math.ceil(out_h) - tf.math.floor(out_h)
    out_w = (in_width - filterSizeW) / strides[1]
    pad_w = tf.math.ceil(out_w) - tf.math.floor(out_w)

    pad_bottom = pad_h * strides[0]
    pad_right = pad_w * strides[1]

    paddings = [[0, 0], [0, pad_bottom], [0, pad_right], [0, 0]]

    padded = tf.pad(input, paddings, mode='CONSTANT',
                    constant_values=-math.inf)
    return (padded, [0, pad_bottom, 0, pad_right])


def _pad_input(input, strides, filterSizeH, filterSizeW, padding, ceil_mode):
    pads = tf.zeros([4])
    # check for explicit padding
    if type(padding) is list:
        input, pads_ = _pad_input_explicit(input, padding)
        pads += pads_
    elif padding[:4].lower() == "same":
        input, pads_ = _pad_input_same(input, strides, filterSizeH,
                                       filterSizeW, padding)
        pads += pads_

    # when padding is set to SAME, ceil_mode will not do anything
    # because output sizes will be multiple of the strides
    if ceil_mode and (type(padding) is list or padding[:4].lower() != "same"):
        input, pads_ = _calc_padding_ceil_mode(input, strides,
                                               filterSizeH, filterSizeW)
        pads += pads_
    return (input, tf.cast(pads, tf.int64))


def _calc_dilated_pool(input, ksize, strides, dilation, padding, ceil_mode):
    kH, kW = ksize
    sH, sW = strides
    dH, dW = dilation

    # size of one filter window calculated from the kernel and dilation
    filterSizeH = (kH - 1) * dH + 1
    filterSizeW = (kW - 1) * dW + 1

    input, pads = _pad_input(input, strides, filterSizeH, filterSizeW,
                             padding, ceil_mode)

    # NHWC to NCHW
    inputs_ = tf.transpose(input, [0, 3, 1, 2])
    input_shape = tf.shape(inputs_, out_type=tf.dtypes.int64)

    nH = input_shape[2]
    nW = input_shape[3]

    sizeH = (((nH - filterSizeH) // sH) + 1) * kH
    sizeW = (((nW - filterSizeW) // sW) + 1) * kW

    y = tf.range(sizeH)
    y = _calc_xy(y, kH, dH, sH)

    x = tf.range(sizeW)
    x = _calc_xy(x, kW, dW, sW)
    ind = _calc_indexes(y, x, nH, nW, sizeH, sizeW)

    ind_ = tf.expand_dims(ind, 0)
    ind_ = tf.expand_dims(ind_, 0)
    ind_ = tf.tile(ind_, [input_shape[0], input_shape[1], 1, 1])

    inputs_ = tf.reshape(inputs_, [input_shape[0], input_shape[1],
                                   input_shape[2] * input_shape[3]])
    new_pool = tf.gather_nd(inputs_, ind_, batch_dims=2)
    new_pool = tf.reshape(new_pool, [input_shape[0], input_shape[1],
                                     sizeH, sizeW])
    # To NHWC
    new_pool = tf.transpose(new_pool, [0, 2, 3, 1])

    return (new_pool, sizeW, pads)


def dilated_maxpool_with_argmax(input, ksize, strides, dilation,
                                padding="VALID", ceil_mode=False):
    kH, kW = ksize
    sH, sW = strides
    dH, dW = dilation

    new_pool, sizeW, pads = _calc_dilated_pool(input, ksize, strides,
                                               dilation, padding, ceil_mode)
    kernel = [1] + list(ksize) + [1]
    maxpool, ind = tf.nn.max_pool_with_argmax(new_pool, ksize=kernel,
                                              strides=kernel, padding="VALID")

    input_shape = tf.shape(input, out_type=tf.dtypes.int64)
    in_width = input_shape[2]
    new_ind = _calc_orig_ind(ind, kH, kW, dH, dW, sH, sW,
                             sizeW, in_width, pads)

    return (maxpool, new_ind)


def dilated_maxpool2d(input, ksize, strides, dilation,
                      padding="VALID", ceil_mode=False):
    # size of one filter window calculated from the kernel and dilation
    filterSizeH = (ksize[0] - 1) * dilation[0] + 1
    filterSizeW = (ksize[1] - 1) * dilation[1] + 1

    if type(padding) is list:
        input, _ = _pad_input_explicit(input, padding)
        padding_ = "VALID"
    elif padding.lower() == "same_lower":
        # Tensorflow does not support SAME_LOWER padding
        input, _ = _pad_input_same(input, strides,
                                filterSizeH, filterSizeW, padding)
        padding_ = "VALID"
    elif padding.lower() == "same_upper":
        padding_ = "SAME"
    else:
        padding_ = padding

    # when padding is set to SAME, ceil_mode will not do anything
    # because output sizes will be multiple of the strides
    if ceil_mode and (type(padding) is list or padding[:4].lower() != "same"):
        input, _ = _calc_padding_ceil_mode(input, strides,
                                           filterSizeH, filterSizeW)

    strides = [1] + list(strides) + [1]
    dilation = [1] + list(dilation) + [1]

    input_shape = tf.shape(input, out_type=tf.dtypes.int64)

    filter = tf.zeros(ksize[0] * ksize[1] * input_shape[3])
    filter = tf.reshape(filter, [ksize[0], ksize[1], input_shape[3]])

    maxpool = tf.nn.dilation2d(input=input, filter=filter, strides=strides,
                               rates=dilation, padding=padding_)

    return maxpool


def dilated_maxpool2d_v2(input, ksize, strides, dilation,
                         padding="VALID", ceil_mode=False):
    new_pool, _, _ = _calc_dilated_pool(input, ksize, strides,
                                        dilation, padding, ceil_mode)

    kernel = [1] + list(ksize) + [1]
    maxpool = tf.nn.max_pool2d(new_pool, ksize=kernel,
                               strides=kernel, padding="VALID")
    return maxpool
