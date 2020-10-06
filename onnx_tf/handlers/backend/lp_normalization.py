import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("LpNormalization")
@tf_func(tf.norm)
class LpNormalization(BackendHandler):

  @classmethod
  def get_attrs_processor_param(cls):
    return {
        "default": {
            "axis": -1,
            "p": 2,
            "keepdims": True
        },
        "rename": {
            "p": "ord"
        }
    }

  @classmethod
  def version_1(cls, node, **kwargs):
    input_tensor = kwargs["tensor_dict"][node.inputs[0]]
    tf_norm = cls.make_tensor_from_onnx_node(node, **kwargs)

    return [
        cls.make_tensor_from_onnx_node(
            node,
            tf_func=tf.math.truediv,
            inputs=[input_tensor, tf_norm],
            **kwargs)
    ]
