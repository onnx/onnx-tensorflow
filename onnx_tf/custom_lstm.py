import tensorflow
import numpy as np
from tensorflow.python.ops import rnn_cell_impl

def extract_onnx_lstm_weights(W_weights, R_weights, B_weights):
  '''
  Extract LSTM weights and reorder/resize them to bring them into the correct TensorFlow format.
  ONNX: input, output, forget, cell
  TF:   input, cell, forget, output
  '''
  input_weights = np.split(W_weights, 4)
  input_weights_correct_order = np.concatenate((input_weights[0], input_weights[3], input_weights[2], input_weights[1]))
  recurrent_weights = np.split(R_weights, 4)
  recurrent_weights_correct_order = np.concatenate((recurrent_weights[0], recurrent_weights[3], recurrent_weights[2], recurrent_weights[1]))
  tf_kernel = np.concatenate((np.matrix.transpose(input_weights_correct_order), np.matrix.transpose(recurrent_weights_correct_order)))

  biases = np.split(B_weights, 8)
  input_bias_correct_order = np.concatenate((biases[0], biases[3], biases[2], biases[1]))
  recurrent_bias_correct_order = np.concatenate((biases[4], biases[7], biases[6], biases[5]))
  tf_bias = np.add(input_bias_correct_order, recurrent_bias_correct_order)
  
  return tf_kernel, tf_bias



class OnnxLSTMCell(tensorflow.contrib.rnn.LSTMCell):
  """LSTMCell that allows you to specify the kernel and bias initializers explicitly"""
  
  def __init__(self, num_units,
               use_peepholes=False, cell_clip=None,
               kernel_initializer=None, bias_initializer=None,
               num_proj=None, proj_clip=None,
               num_unit_shards=None, num_proj_shards=None,
               state_is_tuple=True, activation=None,
               reuse=None, name=None):
    super().__init__(num_units=num_units,
                     use_peepholes=use_peepholes, cell_clip=cell_clip,
                     initializer=kernel_initializer,
                     num_proj=num_proj, proj_clip=proj_clip,
                     num_unit_shards=num_unit_shards, num_proj_shards=num_proj_shards,
                     forget_bias=0.0, state_is_tuple=state_is_tuple,
                     activation=activation, reuse=reuse, name=name)

    self._bias_initializer = bias_initializer


  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

    input_depth = inputs_shape[1].value
    h_depth = self._num_units if self._num_proj is None else self._num_proj
    maybe_partitioner = (
        partitioned_variables.fixed_size_partitioner(self._num_unit_shards)
        if self._num_unit_shards is not None
        else None)
    self._kernel = self.add_variable(
        rnn_cell_impl._WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + h_depth, 4 * self._num_units],
        initializer=self._initializer,
        partitioner=maybe_partitioner)
    self._bias = self.add_variable(
        rnn_cell_impl._BIAS_VARIABLE_NAME,
        shape=[4 * self._num_units],
        initializer=
        self._bias_initializer if self._bias_initializer is not None else tf.zeros_initializer(dtype=self.dtype))
    
    if self._use_peepholes:
      self._w_f_diag = self.add_variable("w_f_diag", shape=[self._num_units],
                                         initializer=self._initializer)
      self._w_i_diag = self.add_variable("w_i_diag", shape=[self._num_units],
                                         initializer=self._initializer)
      self._w_o_diag = self.add_variable("w_o_diag", shape=[self._num_units],
                                         initializer=self._initializer)

    if self._num_proj is not None:
      maybe_proj_partitioner = (
          partitioned_variables.fixed_size_partitioner(self._num_proj_shards)
          if self._num_proj_shards is not None
          else None)
      self._proj_kernel = self.add_variable(
          "projection/%s" % _WEIGHTS_VARIABLE_NAME,
          shape=[self._num_units, self._num_proj],
          initializer=self._initializer,
          partitioner=maybe_proj_partitioner)

    self.built = True
