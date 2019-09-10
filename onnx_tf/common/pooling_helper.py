from numpy import inf
import numpy as np
import itertools


def py_maxpool(input, kernel_shape, strides=None, dilations=None,
               padding=None, ceil_mode=False):
    """
        Implementation of MaxPool operation in Python
        Args:
            input:        input N-D data array in NC* format
            kernel_shape: the size of the kernel along each axis
            strides:      stride along each spatial axis
            dilations:    dilations value along each spatial axis of filter
            padding:      padding for the beginning and ending along each
                          spatial axis. `padding` format should be as follow
                          [x1_begin, x2_begin...x1_end, x2_end,...]
            ceil_mode:    whether to use ceil or floor (default) to compute
                          the output shape.
      Return:
            pooled:       output data from max pooling across the input
            ind:          indices of the selected max values from the input
    """

    def _pooling_output_shape(input_size, ksize, stride,
                              dilation, pad, ceil_mode):
        output_size = (input_size + pad - ((ksize - 1) * dilation + 1) +
                       ((stride-1) if ceil_mode else 0)) // stride + 1
        if (pad):
            if ((output_size - 1) * stride >= input_size + pad):
                output_size -= 1
        return output_size

    def _loop_over_output(batch, channel):
        dims = [range(output_sp_shape[d]) for d in range(spatial_size)]
        for counters in itertools.product(*dims):
            input_ranges = []
            for dim in range(spatial_size):
                dim_start = \
                    counters[dim] * strides[dim] - pads[dim * 2]
                dim_end = \
                    min(dim_start + (kernel_shape[dim] - 1) * dilations[dim]
                        + 1, inp_sp_shape[dim])
                while dim_start < 0:
                    dim_start += dilations[dim]

                cur_range = [i for i in range(dim_start,
                                              dim_end, dilations[dim])]
                input_ranges.append(cur_range)
            maxval = -inf
            maxind = -1
            for input_ind in itertools.product(*input_ranges):
                ind = (batch, channel) + input_ind
                val = input[ind]
                if val > maxval:
                    maxval = val
                    ind = 0
                    for i in range(spatial_size):
                        coef = 1
                        for j in range(i+1, spatial_size):
                            coef *= inp_sp_shape[j]
                        ind += input_ind[i] * coef
                    maxind = ind
            ind = (batch, channel) + counters
            out_pool[ind] = maxval
            out_ind[ind] = maxind

    spatial_size = len(kernel_shape)

    input_shape = np.shape(input)
    inp_sp_shape = input_shape[2:]

    batch_size = input_shape[0]
    channels_num = input_shape[1]

    if strides is None:
        strides = kernel_shape

    if dilations is None:
        dilations = [1] * spatial_size

    if padding is None:
        padding = [0] * spatial_size * 2

    pads = []
    pad_along_axis = []
    output_sp_shape = []

    for dim in range(spatial_size):
        pads.append(padding[dim])
        pads.append(padding[dim + spatial_size])
        pad_along_axis.append(padding[dim] + padding[dim + spatial_size])

        input_size = input_shape[dim + 2]
        output_size = \
            _pooling_output_shape(input_size, kernel_shape[dim],
                                  strides[dim], dilations[dim],
                                  pad_along_axis[dim], ceil_mode)
        output_sp_shape.append(output_size)

    out_pool = np.zeros([input_shape[0], input_shape[1]] +
                        output_sp_shape, input.dtype)
    out_ind = np.zeros([input_shape[0], input_shape[1]] +
                       output_sp_shape, np.int64)

    for batch in range(batch_size):
        for channel in range(channels_num):
            _loop_over_output(batch, channel)

    return out_pool, out_ind
