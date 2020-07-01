import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("SequenceErase")
class SequenceErase(BackendHandler):

  @classmethod
  def chk_pos_in_bounds(cls, input_seq, pos):
    """
    Check the position is in-bounds with respect to the sequence.
    Accepted range for 'position' is in [-n, n - 1], where n is the
    number of tensors in 'input_sequence'.

    :param input_seq: input sequence
    :param pos: position of the output tensor

    :return: True if position is in-bounds 
    """
    seq_length = tf.shape(input_seq.to_sparse(), out_type=pos.dtype)[0]

    cond1 = tf.greater_equal(pos, tf.negative(seq_length))
    cond2 = tf.less_equal(pos, seq_length - 1)

    # pos >= -n and pos < n
    return tf.reduce_all(tf.logical_and(cond1, cond2))

  @classmethod
  def version_11(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    input_sequence = tensor_dict[node.inputs[0]]
    seq_length = tf.shape(input_sequence.to_sparse())[0]
    position = tensor_dict[node.inputs[1]] if len(
        node.inputs) == 2 else seq_length - 1

    # check whether position is in-bounds and assert if not
    result = cls.chk_pos_in_bounds(input_sequence, position)
    assert_pos = tf.Assert(tf.equal(result, True), [result])

    with tf.control_dependencies([assert_pos]):
      s1 = input_sequence[:position]
      s2 = input_sequence[position + 1:]
      return [tf.concat([s1, s2], axis=0)]
