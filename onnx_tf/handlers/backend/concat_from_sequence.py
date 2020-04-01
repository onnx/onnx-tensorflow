import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("ConcatFromSequence")
class ConcatFromSequence(BackendHandler):

  @classmethod
  def version_11(cls, node, **kwargs):
    # get the sequence first
    tensor_dict = kwargs["tensor_dict"]
    input_sequence = tensor_dict[node.inputs[0]]

    axis = node.attrs.get("axis")

    sparse_x = input_sequence.to_sparse()
    output_tensor = tf.sparse.to_dense(sparse_x)

    cond_less = lambda i1, i2, i3, axis, o1: tf.less(i1, i2)
    body_concat = lambda i1, i2, i3, axis, o1: [
          i1 + 1, i2, i3, axis, tf.concat([o1, tf.gather(i3, i1)], axis=axis)
    ]

    i_min = 0
    i_max = tf.shape(output_tensor)[0]
    t = tf.gather(output_tensor, 0)
    output_tensor = tf.gather(output_tensor, tf.range(1, i_max))
    i_max = i_max - 1

    _, _, _, _, output_tensor = tf.while_loop(cond_less,
                                body_concat, [i_min, i_max, output_tensor, axis, t],
                                shape_invariants=[
                                    tf.TensorShape(None),
                                    i_max.get_shape(),
                                    output_tensor.get_shape(),
                                    tf.TensorShape(None),
                                    tf.TensorShape(None)
                                ],
                                parallel_iterations=1)

    return [output_tensor]
