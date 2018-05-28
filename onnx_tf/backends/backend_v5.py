"""Backend for running ONNX on Tensorflow

To run this, you will need to have Tensorflow installed as well.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from onnx_tf.backend import TensorflowBackendBase


class TensorflowBackend(TensorflowBackendBase):
  """ Tensorflow Backend for ONNX
  """

  @classmethod
  def handle_reshape(cls, node, input_dict):
    tensor = input_dict[node.inputs[0]]
    shape = tf.cast(input_dict[node.inputs[1]], tf.int64)
    input_shape = tf.shape(tensor, out_type=tf.int64)

    # Extract indicies of the shape paramter where
    # a copy from the original dimension size is needed.
    copy_indices = tf.squeeze(tf.where(tf.equal(shape,
                                                tf.constant(0, dtype=tf.int64))), -1)

    indices_gathered = tf.gather(input_shape, copy_indices)
    indices_scattered = tf.sparse_to_dense(copy_indices,
                                           tf.cast(tf.shape(shape), tf.int64),
                                           indices_gathered)

    # Perform the copy wherever requested (wherever dim_size == 0)
    copied_shape = shape + indices_scattered
    return [tf.reshape(tensor, copied_shape)]