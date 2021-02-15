import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Pad")
@tf_func(tf.pad)
class Pad(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]
    num_dim = len(tensor_dict[node.inputs[0]].get_shape())
    mode = node.attrs.pop("mode", "constant")

    def check_positive(pads):
      p = tf.greater_equal(pads, tf.zeros((1), dtype=pads.dtype))
      r = tf.reduce_all(p)
      return r

    def process_neg_pads(x, paddings, constant_values):
      # Process negative paddings differently since tf.pad
      # doesn't support negative paddings
      # The ONNX logic is similar to tf.slice. So we just
      # need to compute the begins and sizes for slice op

      i_shape = tf.shape(x, out_type=paddings.dtype)
      i_rank = tf.cast(tf.rank(x), paddings.dtype)
      begins = tf.negative(tf.gather(paddings, tf.range(i_rank)))
      ends = i_shape + tf.gather(paddings, tf.range(i_rank, i_rank*2))
      sizes = ends - begins
      result=tf.slice(x, begins, sizes)
      return [result]

    def process_pos_pads(x, paddings, constant_values):

      def _symmetric_pad(i, x):
        paddings_i = tf.map_fn(lambda e: tf.where(i < e, 1, 0), paddings)
        paddings_i = tf.reshape(paddings_i, [num_dim, 2])
        x = tf.pad(x, paddings_i, 'SYMMETRIC')
        return i + 1, x

      # tf requires int32 paddings
      paddings = tf.cast(tf.transpose(tf.reshape(paddings, [2, num_dim])),
                         dtype=tf.int32)

      if mode.lower() == "edge":
        # Tensorflow doesn't support edge mode so we need to implement the
        # np.pad(x, paddings, mode="edge") logic using Tensorflow ops. A
        # while loop is used to go through the tf.pad 'SYMMETRIC' mode to pad
        # one value at a time for both sides and all dimensions.
        paddings = tf.reshape(paddings, [-1])
        max_i = tf.reduce_max(paddings)
        _, x = tf.while_loop(
            lambda i, x: tf.less(i, max_i), _symmetric_pad, [0, x],
            [tf.TensorShape([]), tf.TensorShape(None)])
        return [x]

      return [
          cls.make_tensor_from_onnx_node(
              node, inputs=[x, paddings, mode, constant_values], **kwargs)
      ]

    if cls.SINCE_VERSION < 11:  # for opset 1 and opset 2
      paddings = tf.constant(node.attrs.pop("pads", None), tf.int32)
      constant_values = node.attrs.pop("value", 0.)

    else:  # for opset 11
      paddings = tensor_dict[node.inputs[1]]
      constant_values = tensor_dict[node.inputs[2]] if len(
          node.inputs) == 3 else 0

    cond = tf.cond(check_positive(paddings),
                   lambda: process_pos_pads(x, paddings, constant_values),
                   lambda: process_neg_pads(x, paddings, constant_values))
    return cond

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_2(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
