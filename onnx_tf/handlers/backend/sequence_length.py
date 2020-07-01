import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("SequenceLength")
class SequenceLength(BackendHandler):

  @classmethod
  def version_11(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    input_sequence = tensor_dict[node.inputs[0]]

    return [tf.shape(input_sequence.to_sparse(), out_type=tf.int64)[0]]
