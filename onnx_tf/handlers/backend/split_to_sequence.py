import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.common.tf_helper import tf_shape
from onnx_tf.handlers.handler import partial_support
from onnx_tf.handlers.handler import ps_description

@onnx_op("SplitToSequence")
@partial_support(True)
@ps_description("Scalar as the split input not supported.")
class SplitToSequence(BackendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    axis = node.attrs.get("axis", 0)
    input_shape = kwargs["tensor_dict"][node.inputs[0]].get_shape()
    x_rank = len(input_shape) 
    if axis > x_rank - 1 or axis < -x_rank:
        raise ValueError("Axis is out of bound")

  @classmethod
  def version_11(cls, node, **kwargs):
    # split the input first
    tensor_dict = kwargs["tensor_dict"]
    dtype = tensor_dict[node.inputs[0]].dtype
    original_input = tensor_dict[node.inputs[0]]
    split = tensor_dict[node.inputs[1]] if len(node.inputs) > 1 else None
    axis = node.attrs.get("axis", 0)
    keepdims = node.attrs.get("keepdims", 1)
    input_shape = tf_shape(original_input)

    if len(node.inputs) > 1:
      split_shape = tf_shape(split)
      # check if the split is 1-d or scalar
      if split_shape.shape[0] == 1:
        split_sizes = split
      else:
        # Need to build the split sizes
        # First int(size/n) of ns [n, n, n...]
        # Then append m if needed [n, n, n..., m] where m=size(mod n)
        # Currently tf.split does not take an unknown shape tensor
        # for the num_or_size_splits input. Since this parameter
        # has to be calculated based on ONNX inputs, the shape is
        # unknown during graph generation time, causing a Tensorflow
        # exception.
        # Due to the limitation in tf.split, this option is currently
        # not supported.
        # split_sizes = tf.tile([split], tf.reshape(tf.math.floordiv(
        #    tf.cast(input_shape[axis], dtype=tf.int32), split), [1]))
        raise RuntimeError(
          "Split to sequence with scalar split is not supported due to API limitations.")
      split_inputs = tf.split(original_input, split_sizes, axis=axis)

    else:
      # split is not provided, use default 1
      split_sizes = tf.tile([1], tf.reshape(input_shape[axis], [1]))
      split_inputs = tf.split(original_input, split_sizes, axis=axis)
      if keepdims == 0:
        split_inputs = [tf.squeeze(split_input) for split_input in split_inputs]

    # create an empty sequence next
    input_sequence = tf.ragged.constant([], dtype=dtype)

    # insert tensors at the end of sequence
    for i in range(len(split_inputs)):
      input_tensor = tf.expand_dims(split_inputs[i], 0)
      if input_sequence.shape[0] == 0:
        output_seq = tf.RaggedTensor.from_tensor(input_tensor)
      else:
        output_seq = tf.concat([input_sequence, input_tensor], axis=0)
      input_sequence = output_seq

    return [output_seq]
