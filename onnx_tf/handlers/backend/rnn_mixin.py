from functools import partial

import tensorflow as tf
from tensorflow.python.ops import array_ops

from onnx_tf.common import ONNX_OP_TO_TF_OP
from onnx_tf.common import EXPERIMENTAL_ONNX_OP_TO_TF_OP


class RNNMixin(object):

  @classmethod
  def rnn(cls, x, cell_class, cell_kwargs, rnn_kwargs, activations, direction):
    cell_kwargs["activation"] = activations[0]

    rnn_cell = [cell_class(**cell_kwargs)]
    cell_fw = tf.nn.rnn_cell.MultiRNNCell(rnn_cell)

    if direction == "bidirectional":
      cell_kwargs["activation"] = activations[1]
      rnn_cell_bw = [cell_class(**cell_kwargs)]
      cell_bw = tf.nn.rnn_cell.MultiRNNCell([rnn_cell_bw])

    if direction == "forward":
      outputs, states = tf.nn.dynamic_rnn(cell_fw, x, **rnn_kwargs)
    elif direction == "bidirectional":
      outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x,
                                                        **rnn_kwargs)
    elif direction == "reverse":

      def _reverse(input_, seq_dim):
        return array_ops.reverse(input_, axis=[seq_dim])

      time_dim = 0
      inputs_reverse = _reverse(x, time_dim)
      outputs, states = tf.nn.dynamic_rnn(cell_fw, inputs_reverse, **rnn_kwargs)
      outputs = _reverse(outputs, time_dim)

    return outputs, states

  @classmethod
  def rnn_get_activation(cls, name, alpha, beta):
    op_dict = ONNX_OP_TO_TF_OP.copy()
    op_dict.update(EXPERIMENTAL_ONNX_OP_TO_TF_OP)
    if name not in op_dict:
      raise NotImplementedError(
          "Activation function {} is not supported.".format(name))
    activation = op_dict[name]
    kwargs = {}
    if name == "affine":
      kwargs["scale"] = alpha
      kwargs["shift"] = beta
      activation = activation(**kwargs)
    elif name == "elu":
      assert alpha == 1, "TensorFlow does not support alpha, else 1."
    elif name == "hard_sigmoid":
      assert alpha == 0.2, "TensorFlow can only set default alpha 0.2."
      assert beta == 0.5, "TensorFlow can only set default beta 0.5"
    elif name == "leaky_relu":
      kwargs["alpha"] = alpha or 0.01
      activation = partial(activation, **kwargs)
    elif name == "thresholded_relu":
      kwargs["theta"] = alpha
      activation = activation(**kwargs)
    return activation
