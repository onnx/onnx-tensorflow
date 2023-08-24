import numpy as np
import tensorflow as tf


class PadMixin(object):

  @classmethod
  def get_padding_as_op(cls, x, pads, format:str=None):
    num_dim = int(len(pads) / 2)

    # tf_pads = np.transpose(np.array(pads).reshape([2, num_dim]))
    if format is None:
      NIdx, CIdx = 0, 1
    else:
      assert "N" in format and "C" in format, "expected `N` and `C` in padding op's input format " \
                                              "if given"
      NIdx = format.index("N")
      CIdx = format.index("C")
    # create an empty tf_pads array
    tf_pads = np.zeros([num_dim + 2, 2], dtype=np.int32)
    # the indices of spatial axes in input format.
    spatial_indices = [axis for axis in range(num_dim + 2) if axis not in [NIdx, CIdx]]
    # fill pads into tf_pads's spatial axes
    tf_pads[spatial_indices, :] = np.transpose(np.array(pads).reshape([2, num_dim]))

    padding = tf.constant(tf_pads)  # tf requires int32 paddings
    return tf.pad(x, padding)