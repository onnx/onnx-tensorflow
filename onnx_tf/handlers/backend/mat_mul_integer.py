import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("MatMulInteger")
@tf_func(tf.matmul)
class MatMulInteger(BackendHandler):

  @classmethod
  def version_10(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    A = tensor_dict[node.inputs[0]]
    B = tensor_dict[node.inputs[1]]
    # tf.matmul doesn't support int8 and uint8 for A and B,
    # therefore need to cast them to int32
    A = tf.cast(A, tf.int32)
    B = tf.cast(B, tf.int32)

    if 'a_zero_point' in tensor_dict:
      a_zero_point = tensor_dict['a_zero_point']
      shape = a_zero_point.get_shape().as_list()
      if len(shape) > 0 and shape[0] > 1:
        # reshape a_zero_point before subtract it from A
        a_zero_point = tf.reshape(a_zero_point, [shape[0], 1])
      a_zero_point = tf.cast(a_zero_point, tf.int32)
      A = tf.subtract(A, a_zero_point)

    if 'b_zero_point' in tensor_dict:
      b_zero_point = tensor_dict['b_zero_point']
      b_zero_point = tf.cast(b_zero_point, tf.int32)
      B = tf.subtract(B, b_zero_point)

    return [cls.make_tensor_from_onnx_node(node, inputs=[A, B], **kwargs)]
