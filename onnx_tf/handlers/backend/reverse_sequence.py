import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("ReverseSequence")
@tf_func(tf.reverse_sequence)
class ReverseSequence(BackendHandler):

  @classmethod
  def get_attrs_processor_param(cls):
    return {
        "default": {
            "time_axis": 0,
            "batch_axis": 1
        },
        "rename": {
            "time_axis": "seq_axis"
        }
    }

  @classmethod
  def version_10(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]
