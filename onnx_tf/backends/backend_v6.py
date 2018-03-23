"""Backend for running ONNX on Tensorflow

To run this, you will need to have Tensorflow installed as well.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx_tf.backend import TensorflowBackendBase
import tensorflow as tf

class TensorflowBackend(TensorflowBackendBase):
  """ Tensorflow Backend for ONNX
  """

  @classmethod
  def handle_batch_normalization(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    total_num_dim = len(x.get_shape())
    scale = cls._explicit_broadcast(input_dict[node.inputs[1]], 1, total_num_dim)
    bias = cls._explicit_broadcast(input_dict[node.inputs[2]], 1, total_num_dim)
    running_mean = cls._explicit_broadcast(input_dict[node.inputs[3]], 1, total_num_dim)
    running_variance = cls._explicit_broadcast(input_dict[node.inputs[4]], 1, total_num_dim)

    variance_epsilon = node.attrs.get("epsilon", 0.00001)
    if node.attrs.get("is_test", 0):
      return [tf.nn.batch_normalization(x, running_mean, running_variance, bias, scale,
                                        variance_epsilon)]
    spatial = node.attrs.get("spatial", 1) == 1
    momentum = node.attrs.get("momentum", 0.9)
    axis = [0] if spatial else [0] + list(range(2, total_num_dim))
    mean, variance = tf.nn.moments(x, axis)
    mean = cls._explicit_broadcast(mean, 1, total_num_dim)
    variance = cls._explicit_broadcast(variance, 1, total_num_dim)
    running_mean = running_mean * momentum + mean * (1 - momentum)
    running_variance = running_variance * momentum + variance * (1 - momentum)
    # TODO: need to conform to the documentation here
    return [tf.nn.batch_normalization(x, running_mean, running_variance, bias, scale,
                                      variance_epsilon)]
