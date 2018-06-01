import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("RandomUniformLike")
@tf_func(tf.random_uniform)
class RandomUniformLike(BackendHandler):

  @classmethod
  def process_attrs(cls, attrs):
    return cls._process_attrs(
        attrs,
        rename={
            "minval": "low",
            "maxval": "high"
        },
        default={
            "high": 1.,
            "low": 0.
        })

  @classmethod
  def version_1(cls, node, **kwargs):
    inputs = [kwargs["tensor_dict"][node.inputs[0]].get_shape()]
    return [cls.make_tf_tensor(node, inputs=inputs, **kwargs)]
