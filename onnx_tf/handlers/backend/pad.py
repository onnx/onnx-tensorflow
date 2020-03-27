import numpy as np
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

    def process_neg_pads(x, paddings):
      # process negative paddings differently since TF.pad
      # doesn't support negative paddings

      i_shape = tf.shape(x)
      cond_less = lambda i1, i2, o1: tf.less(i1, i2)
      body_concat = lambda i1, i2, o1: [
          i1 + 1, i2, tf.concat([o1, [i1]], axis=0)
      ]
      cond_neg_pads = lambda i1, i2, i3, o1: tf.less(i1, i2)

      def _loop_neg_pads(i, i_x, p, result):
        # process one dimension at a time

        i_min = tf.negative(tf.gather(p, i * 2))
        i_max = i_shape[i] + tf.gather(p, i * 2 + 1)
        t = tf.constant([0])
        _, _, r = tf.while_loop(cond_less,
                                body_concat, [i_min, i_max, t],
                                shape_invariants=[
                                    i_min.get_shape(),
                                    i_max.get_shape(),
                                    tf.TensorShape([None])
                                ],
                                parallel_iterations=1)
        gather_indices = tf.gather(r, tf.range(1, tf.size(r)))
        result = tf.gather(result, gather_indices)

        # prepare for the next loop
        i_min = tf.constant(0)
        i_max = i_x
        _, _, r = tf.while_loop(cond_less,
                                body_concat, [i_min, i_max, t],
                                shape_invariants=[
                                    i_min.get_shape(),
                                    i_max.get_shape(),
                                    tf.TensorShape([None])
                                ],
                                parallel_iterations=1)
        transpose_indices = tf.gather(r, tf.range(1, tf.size(r)))
        transpose_indices = tf.roll(transpose_indices, shift=-1, axis=0)
        result = tf.transpose(result, transpose_indices)
        return i + 1, i_x, p, result

      # tf requires int32 paddings
      paddings = tf.cast(paddings, dtype=tf.int32)
      i = tf.constant(0)
      i_rank = tf.rank(x)
      _, _, _, result = tf.while_loop(cond_neg_pads,
                                      _loop_neg_pads, [i, i_rank, paddings, x],
                                      shape_invariants=[
                                          i.get_shape(),
                                          i_rank.get_shape(),
                                          paddings.get_shape(),
                                          tf.TensorShape(None)
                                      ],
                                      parallel_iterations=1)
      return [result]

    def process_pos_pads(x, paddings):

      def _symmetric_pad(i, x):
        paddings_i = tf.map_fn(lambda e: tf.where(i < e, 1, 0), paddings)
        paddings_i = tf.reshape(paddings_i, [num_dim, 2])
        x = tf.pad(x, paddings_i, 'SYMMETRIC')
        return i + 1, x

      # tf requires int32 paddings
      paddings = tf.cast(tf.transpose(tf.reshape(paddings, [2, num_dim])),
                         dtype=tf.int32)

      if mode.lower() == "edge":
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
      paddings = node.attrs.pop("pads", None)
      constant_values = node.attrs.pop("value", 0.)

    else:  # for opset 11
      paddings = tensor_dict[node.inputs[1]]
      constant_values = tensor_dict[node.inputs[2]] if len(
          node.inputs) == 3 else 0

    cond = tf.cond(check_positive(paddings),
                   lambda: process_pos_pads(x, paddings),
                   lambda: process_neg_pads(x, paddings))
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
