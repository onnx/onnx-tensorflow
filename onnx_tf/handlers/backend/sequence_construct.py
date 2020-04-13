import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("SequenceConstruct")
class SequenceConstruct(BackendHandler):

  @classmethod
  def version_11(cls, node, **kwargs):
    # create an empty sequence first
    tensor_dict = kwargs["tensor_dict"]
    dtype = tensor_dict[node.inputs[0]].dtype
    input_sequence = tf.ragged.constant([], dtype=dtype)

    # insert tensors at the end of sequence
    for i in range(len(node.inputs)):
      input_tensor = tf.expand_dims(tensor_dict[node.inputs[i]], 0)
      if input_sequence.shape[0] == 0:
        output_seq = tf.RaggedTensor.from_tensor(input_tensor)
      else:
        output_seq = tf.concat([input_sequence, input_tensor], axis=0)
      input_sequence = output_seq

    return [output_seq]
