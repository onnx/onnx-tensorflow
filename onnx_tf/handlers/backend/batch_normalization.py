import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("BatchNormalization")
@tf_func(tf.nn.batch_normalization)
class BatchNormalization(BackendHandler):

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
    x = tensor_dict[node.inputs[0]]
    x_shape = x.get_shape().as_list()
    x_rank = len(x_shape)

    params_shape_broadcast = list(
        [1, x_shape[1]] + [1 for _ in range(2, x_rank)])

    total_num_dim = len(x.get_shape())
    scale = tf.reshape(tensor_dict[node.inputs[1]], params_shape_broadcast)
    bias = tf.reshape(tensor_dict[node.inputs[2]], params_shape_broadcast)
    running_mean = tf.reshape(tensor_dict[node.inputs[3]],
                              params_shape_broadcast)
    running_variance = tf.reshape(tensor_dict[node.inputs[4]],
                                  params_shape_broadcast)

    # from version 7, force to use test mode
    if cls.SINCE_VERSION >= 7 or node.attrs.get("is_test", 0):
      inputs = [x, running_mean, running_variance, bias, scale]
      return [cls.make_tensor_from_onnx_node(node, inputs=inputs)]
    spatial = node.attrs.get("spatial", 1) == 1
    momentum = node.attrs.get("momentum", 0.9)
    axis = [0] if spatial else [0] + list(range(2, total_num_dim))
    mean, variance = tf.nn.moments(x, axis)
    running_mean = running_mean * momentum + mean * (1 - momentum)
    running_variance = running_variance * momentum + variance * (1 - momentum)
    # TODO: need to conform to the documentation here
    inputs = [x, running_mean, running_variance, bias, scale]
    return [cls.make_tensor_from_onnx_node(node, inputs=inputs)]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_9(cls, node, **kwargs):
    return cls._common(node, **kwargs)
