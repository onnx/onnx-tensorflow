import tensorflow as tf
import numpy as np


def tf_shape(tensor):
    """
        Helper function returning the shape of a Tensor.
        The function will check for fully defined shape and will return
        numpy array or if the shape is not fully defined will use tf.shape()
        to return the shape as a Tensor.
    """
    if tensor.shape.is_fully_defined():
        return np.array(tensor.shape.as_list(), dtype=np.int64)
    else:
        return tf.shape(tensor, out_type=tf.int64)


def tf_product(a, b):
    """
        Calculates the cartesian product of two column vectors a and b

        Example:

        a = [[1]
             [2]
             [3]]

        b = [[0]
             [1]]

        result = [[1 0]
                  [1 1]
                  [2 0]
                  [2 1]
                  [3 0]
                  [3 1]]
    """
    tile_a = tf.tile(a, [1, tf.shape(b)[0]])
    tile_a = tf.expand_dims(tile_a, 2)
    tile_a = tf.reshape(tile_a, [-1, 1])

    b = tf.tile(b, [tf.shape(a)[0], 1])
    b = tf.concat([tile_a, b], axis=1)

    return b
