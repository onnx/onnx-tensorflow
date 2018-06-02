import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("RandomUniform")
@tf_func(tf.random_uniform)
class RandomUniform(BackendHandler):

  @classmethod
  def process_attrs(cls, attrs):
    return cls._process_attrs(
        attrs,
        rename={
            "low": "minval",
            "high": "maxval"
        },
        default={
            "low": 0.,
            "high": 1.
        })

  @classmethod
  def version_1(cls, node, **kwargs):
    return [cls.make_tf_tensor(node, **kwargs)]
