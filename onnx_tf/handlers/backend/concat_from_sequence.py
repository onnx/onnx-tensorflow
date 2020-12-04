import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import partial_support
from onnx_tf.handlers.handler import ps_description

@onnx_op("ConcatFromSequence")
@partial_support(True)
@ps_description("new_axis=1 not supported in Tensorflow.")
class ConcatFromSequence(BackendHandler):

  @classmethod
  def version_11(cls, node, **kwargs):
    # get the sequence and convert to a tensor
    tensor_dict = kwargs["tensor_dict"]
    input_sequence = tensor_dict[node.inputs[0]]
    output_tensor = tf.sparse.to_dense(input_sequence.to_sparse())
    i_min = 0
    i_max = tf.shape(output_tensor)[0]

    axis = node.attrs.get("axis")
    new_axis = node.attrs.get("new_axis", 0)

    if new_axis == 1:
      # Currently this case, np.stack like behavior, is not supported.
      # The commmented code below would work if tf.unstack supports
      # a scalar tensor input for num.
      # tensor_list = tf.unstack(output_tensor, num=i_max)
      # output_tensor = tf.stack(tensor_list, axis)
      raise RuntimeError(
          "Concat from sequence with new_axis=1 not supported in Tensorflow.")
    else:
      # define the condition and body for the while loop
      cond_less = lambda i1, i2, i3, axis, o1: tf.less(i1, i2)
      body_concat = lambda i1, i2, i3, axis, o1: [
          i1 + 1, i2, i3, axis,
          tf.concat([o1, tf.gather(i3, i1)], axis=axis)
      ]

      # initialize with the first element
      t = tf.gather(output_tensor, 0)

      # setup inputs for the while loop
      input_tensor = tf.gather(output_tensor, tf.range(1, i_max))
      i_max = i_max - 1

      # loop through the rest of elements
      _, _, _, _, output_tensor = tf.while_loop(
          cond_less,
          body_concat, [i_min, i_max, input_tensor, axis, t],
          shape_invariants=[
              tf.TensorShape(None),
              i_max.get_shape(),
              input_tensor.get_shape(),
              tf.TensorShape(None),
              tf.TensorShape(None)
          ],
          parallel_iterations=1)

    return [output_tensor]
