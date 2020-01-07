import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("RandomUniformLike")
@tf_func(tf.random.uniform)
class RandomUniformLike(BackendHandler):

  @classmethod
  def get_attrs_processor_param(cls):
    return {
        "default": {
            "low": 0.,
            "high": 1.
        },
        "rename": {
            "low": "minval",
            "high": "maxval"
        }
    }

  @classmethod
  def version_1(cls, node, **kwargs):
    inputs = [kwargs["tensor_dict"][node.inputs[0]].get_shape()]
    return [cls.make_tensor_from_onnx_node(node, inputs=inputs, **kwargs)]
