import math
import numpy as np


def py_maxpool(input, ksize, strides, dilation=[1, 1],
               pads=[0, 0, 0, 0], ceil_mode=False):
    """
        Implementation of MaxPool operation in Python
        Args:
            input:      input 4D data array in NCHW format
            ksize:      the size of the kernel along each axis
            strides:    stride along each spatial axis
            dilation:   dilation value along each spatial axis of filter
            pads:       padding for the beginning and ending along each
                        spatial axis. `pads` format should be as follow
                        [x1_begin, x2_begin...x1_end, x2_end,...]
            ceil_mode:  wether to use ceil or floor (default) to compute
                        the output shape.
      Return:
            pooled:     output data from max pooling across the input
            ind:        indices from max pooling across the input
    """

    def _pooling_output_shape(input_size, ksize, stride,
                              dilation, pad, ceil_mode):
        output_size = (input_size + pad - ((ksize - 1) * dilation + 1) +
                       ((stride-1) if ceil_mode else 0)) // stride + 1
        if (pad):
            if ((output_size - 1) * stride >= input_size + pad):
                output_size -= 1
        return output_size

    kH, kW = ksize
    sH, sW = strides
    dH, dW = dilation

    pad_top, pad_left, pad_bottom, pad_right = pads
    padH = pad_top + pad_bottom
    padW = pad_left + pad_right

    input_shape = np.shape(input)
    iheight, iwidth = input_shape[2:4]

    oheight = _pooling_output_shape(iheight, kH, sH, dH, padH, ceil_mode)
    owidth = _pooling_output_shape(iwidth, kW, sW, dW, padW, ceil_mode)

    out_pool = np.zeros((input_shape[0], input_shape[1],
                        oheight, owidth), input.dtype)
    out_ind = np.zeros((input_shape[0], input_shape[1],
                       oheight, owidth), 'int64')

    for batch in range(input_shape[0]):
        for channel in range(input_shape[1]):
            # Loop over output
            for i in range(oheight):
                for j in range(owidth):
                    hstart = i * sH - pad_top
                    wstart = j * sW - pad_left
                    hend = min(hstart + (kH - 1) * dH + 1, iheight)
                    wend = min(wstart + (kW - 1) * dW + 1, iwidth)
                    while hstart < 0:
                        hstart += dH
                    while wstart < 0:
                        wstart += dW

                    maxind = -1
                    maxval = -math.inf

                    for y in range(hstart, hend, dH):
                        for x in range(wstart, wend, dW):
                            val = input[batch][channel][y][x]
                            if val > maxval:
                                maxval = val
                                maxind = y * iwidth + x
                    out_pool[batch][channel][i][j] = maxval
                    out_ind[batch][channel][i][j] = maxind

    return (out_pool, out_ind)
