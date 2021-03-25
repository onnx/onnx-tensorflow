import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("InstanceNormalization")
@tf_func(tf.nn.batch_normalization)
class InstanceNormalization(BackendHandler):

  @classmethod
  def get_attrs_processor_param(cls):
    return {
        "default": {
            "epsilon": 1e-5
        },
        "rename": {
            "epsilon": "variance_epsilon"
        }
    }

  @classmethod
  def _common(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    # this file is adapted from :
    # https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/layers/python/layers/normalization.py.
    # We do not use the tf layer instance_norm because there is no way
    # to pass in tensor as beta or gamma.
    gamma = tensor_dict[node.inputs[1]]
    beta = tensor_dict[node.inputs[2]]

    inputs = tensor_dict[node.inputs[0]]
    inputs_shape = inputs.shape
    inputs_rank = inputs.shape.ndims

    moments_axes = list(range(inputs_rank))[2:]
    channel_size = inputs_shape[
        1] if inputs_shape[1] is not None else gamma.shape[0]
    params_shape_broadcast = list([1, channel_size] +
                                  [1 for _ in range(2, inputs_rank)])

    beta = tf.reshape(beta, params_shape_broadcast)
    gamma = tf.reshape(gamma, params_shape_broadcast)

    # Calculate the moments (instance activations).
    mean, variance = tf.nn.moments(inputs, moments_axes, keepdims=True)

    # Compute instance normalization.
    inputs = [inputs, mean, variance, beta, gamma]
    return [
        cls.make_tensor_from_onnx_node(node,
                                       inputs=inputs,
                                       name="instancenorm",
                                       **kwargs)
    ]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls._common(node, **kwargs)
