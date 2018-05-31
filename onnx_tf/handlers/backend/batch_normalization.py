import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.handlers.handler import tf_op


@onnx_op("BatchNormalization")
@tf_op(["BatchNorm", "FusedBatchNorm"])
@tf_func(tf.nn.batch_normalization)
class BatchNormalization(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]
    total_num_dim = len(x.get_shape())
    scale = cls._explicit_broadcast(tensor_dict[node.inputs[1]], 1,
                                    total_num_dim)
    bias = cls._explicit_broadcast(tensor_dict[node.inputs[2]], 1, total_num_dim)
    running_mean = cls._explicit_broadcast(tensor_dict[node.inputs[3]], 1,
                                           total_num_dim)
    running_variance = cls._explicit_broadcast(tensor_dict[node.inputs[4]], 1,
                                               total_num_dim)

    variance_epsilon = node.attrs.get("epsilon", 0.00001)
    if node.attrs.get("is_test", 0):
      inputs = [x, running_mean, running_variance, bias, scale, variance_epsilon]
      return [cls.make_tf_tensor(node, inputs=inputs)]
    spatial = node.attrs.get("spatial", 1) == 1
    momentum = node.attrs.get("momentum", 0.9)
    axis = [0] if spatial else [0] + list(range(2, total_num_dim))
    mean, variance = tf.nn.moments(x, axis)
    mean = cls._explicit_broadcast(mean, 1, total_num_dim)
    variance = cls._explicit_broadcast(variance, 1, total_num_dim)
    running_mean = running_mean * momentum + mean * (1 - momentum)
    running_variance = running_variance * momentum + variance * (1 - momentum)
    # TODO: need to conform to the documentation here
    inputs = [x, running_mean, running_variance, bias, scale, variance_epsilon]
    return [cls.make_tf_tensor(node, inputs=inputs)]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls._common(node, **kwargs)
