from functools import partial

import tensorflow as tf

from onnx_tf.common import get_unique_suffix
from onnx_tf.common import exception
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from .rnn_mixin import RNNMixin
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class GRUCellWithLinearBeforeReset(tf.contrib.rnn.LayerRNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
    dtype: Default dtype of the layer (default of `None` means use the type
      of the first input). Required when `build` is called before `call`.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None,
               dtype=None):
    super(GRUCellWithLinearBeforeReset, self).__init__(_reuse=reuse, name=name, dtype=dtype)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value
    self._gate_kernel = self.add_variable(
        "gates/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, 2 * self._num_units],
        initializer=self._kernel_initializer)
    self._gate_bias = self.add_variable(
        "gates/%s" % _BIAS_VARIABLE_NAME,
        shape=[2 * self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.constant_initializer(1.0, dtype=self.dtype)))
    self._candidate_bias_rbh = self.add_variable(
        "candidate_rbh/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.zeros_initializer(dtype=self.dtype)))
    self._candidate_bias_wbh = self.add_variable(
        "candidate_wbh/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.zeros_initializer(dtype=self.dtype)))
    self._candidate_kernel_rh = self.add_variable(
        "candidate_rh/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[self._num_units, self._num_units],
        initializer=self._kernel_initializer)
    self._candidate_kernel_wh = self.add_variable(
        "candidate_wh/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth, self._num_units],
        initializer=self._kernel_initializer)

    self.built = True

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""

    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, state], 1), self._gate_kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

    value = math_ops.sigmoid(gate_inputs)
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    b_in, b_hn = (self._candidate_bias_rbh, self._candidate_bias_wbh)

    linear_gate_state = math_ops.matmul(state, self._candidate_kernel_rh)
    linear_gate_state = nn_ops.bias_add(linear_gate_state, self._candidate_bias_rbh)
    r_state = r * linear_gate_state

    candidate = math_ops.matmul(inputs, self._candidate_kernel_wh)
    candidate = nn_ops.bias_add(candidate, self._candidate_bias_wbh)

    c = self._activation(candidate + r_state)
    new_h = u * state + (1 - u) * c
    return new_h, new_h


@onnx_op("GRU")
class GRU(RNNMixin, BackendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    direction = node.attrs.get("direction", "forward")
    num_directions = 2 if direction == "bidirectional" else 1
    if "clip" in node.attrs:
      exception.OP_UNSUPPORTED_EXCEPT("GRU with clip", "Tensorflow")
    if "activations" in node.attrs:
      activations = list(map(lambda x: x.lower(), node.attrs["activations"]))
      if activations[0] != "sigmoid":
        exception.OP_UNSUPPORTED_EXCEPT("GRU without sigmoid for `z` and `r`",
                                        "Tensorflow")
      if num_directions == 2:
        if activations[2] != "sigmoid":
          exception.OP_UNSUPPORTED_EXCEPT("GRU without sigmoid for `z` and `r`",
                                          "Tensorflow")

  @classmethod
  def _custom_getter(cls,
                     getter,
                     name,
                     node=None,
                     tensor_dict=None,
                     is_bidirectional=None,
                     *args,
                     **kwargs):
    names = name.split("/")
    if is_bidirectional:
      if "fw" in names:
        index = 0
      elif "bw" in names:
        index = 1
      else:
        raise RuntimeError("Can not get {} for bidirectional. "
                           "Either fw and bw is not in name scope.".format(
                               names[-1]))
    if names[-1] == "kernel":
      # onnx W[zrh], R[zrh]
      if is_bidirectional:
        w = tf.split(tensor_dict[node.inputs[1]], 2)[index]
        r = tf.split(tensor_dict[node.inputs[2]], 2)[index]
      else:
        w = tensor_dict[node.inputs[1]]
        r = tensor_dict[node.inputs[2]]
      w_z, w_r, w_h = tf.split(tf.squeeze(w), 3)
      r_z, r_r, r_h = tf.split(tf.squeeze(r), 3)
      if names[-2] == "gates":
        new_w = tf.transpose(tf.concat([w_r, w_z], 0))
        new_r = tf.transpose(tf.concat([r_r, r_z], 0))
      elif names[-2] == "candidate" or names[-2] == "candidate_rh" or names[-2] == "candidate_wh":
        new_w = tf.transpose(w_h)
        new_r = tf.transpose(r_h)
      if names[-2] == "candidate_rh":
          return new_r
      elif names[-2] == "candidate_wh":
        return new_w
      else:
        kernel = tf.concat([new_w, new_r], 0)
        return kernel
    if names[-1] == "bias":
      if len(node.inputs) >= 4:
        # onnx Wb[zrh], Rb[zrh]
        if is_bidirectional:
          b = tf.split(tensor_dict[node.inputs[3]], 2)[index]
        else:
          b = tensor_dict[node.inputs[3]]
        w_b, r_b = tf.split(tf.squeeze(b), 2)
        w_b_z, w_b_r, w_b_h = tf.split(w_b, 3)
        r_b_z, r_b_r, r_b_h = tf.split(r_b, 3)
        if names[-2] == "gates":
          w_b = tf.transpose(tf.concat([w_b_r, w_b_z], 0))
          r_b = tf.transpose(tf.concat([r_b_r, r_b_z], 0))
        elif names[-2] == "candidate" or names[-2] == "candidate_rbh" or names[-2] == "candidate_wbh":
          w_b = tf.transpose(w_b_h)
          r_b = tf.transpose(r_b_h)
        if names[-2] == "candidate_rbh":
          return r_b
        elif names[-2] == "candidate_wbh":
          return w_b
        else:
          return tf.add(w_b, r_b)
      return getter(name, *args, **kwargs)
    return getter(name, *args, **kwargs)

  @classmethod
  def _common(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]
    input_shape = x.get_shape().as_list()
    input_size = len(node.inputs)
    hidden_size = node.attrs["hidden_size"]
    direction = node.attrs.get("direction", "forward")
    num_directions = 2 if direction == "bidirectional" else 1

    # removed from version 7, default is 0
    output_sequence = node.attrs.get("output_sequence", 0)

    # TODO(fumihwh): check if prev node is one of RNN
    # process input if it comes from other previous cell
    # which has shape [seq_length, num_directions, batch_size, hidden_size]
    if len(input_shape) == 4 and input_shape[1] == 1:
      x = tf.squeeze(x, axis=[1])

    sequence_length = None
    if input_size >= 5 and node.inputs[4] in tensor_dict:
      sequence_length = tensor_dict[node.inputs[4]]

    cell_kwargs = {}

    tf_activations = [tf.nn.tanh] * num_directions
    if "activations" in node.attrs:
      activations = list(map(lambda x: x.lower(), node.attrs["activations"]))
      activation_alpha = node.attrs.get("activation_alpha", [None] * 4)
      activation_beta = node.attrs.get("activation_beta", [None] * 4)
      tf_activations = [
          cls.rnn_get_activation(activations[1], activation_alpha[1],
                                 activation_beta[1])
      ]
      if num_directions == 2:
        tf_activations.append(
            cls.rnn_get_activation(activations[3], activation_alpha[3],
                                   activation_beta[3]))

    # TODO(fumihwh): check if reverse and bidirectional works
    with tf.variable_scope(
        "GRU_" + get_unique_suffix(),
        custom_getter=partial(
            cls._custom_getter,
            node=node,
            tensor_dict=tensor_dict,
            is_bidirectional=num_directions == 2)):

      cell_kwargs["num_units"] = hidden_size
      if input_size < 4 or node.inputs[3] not in tensor_dict:
        cell_kwargs["bias_initializer"] = tf.zeros_initializer
      initial_state = None
      initial_state_bw = None
      if input_size == 6:
        initial_h = tensor_dict.get(node.inputs[5], None)
        if initial_h is not None:
          initial_state = (initial_h[0],)
          if num_directions == 2:
            initial_state_bw = (initial_h[1],)

      rnn_kwargs = {}
      if num_directions == 1:
        rnn_kwargs["initial_state"] = initial_state
      elif num_directions == 2:
        rnn_kwargs["initial_state_fw"] = initial_state
        rnn_kwargs["initial_state_bw"] = initial_state_bw
      rnn_kwargs["sequence_length"] = sequence_length
      rnn_kwargs["time_major"] = True
      rnn_kwargs["dtype"] = tf.float32

      if node.attrs.get("linear_before_reset", 0):
        cell_class = GRUCellWithLinearBeforeReset
      else:
        cell_class = tf.nn.rnn_cell.GRUCell

      outputs, states = cls.rnn(x, cell_class, cell_kwargs,
                                rnn_kwargs, tf_activations, direction)

    if num_directions == 1:
      state = states[0]
      h = tf.expand_dims(state, 0)
      # output = tf.expand_dims(outputs, 1)
      output = outputs
    else:
      state_fw = states[0][0]
      state_bw = states[1][0]
      output_fw = outputs[0]
      output_bw = outputs[1]
      h_fw = tf.expand_dims(state_fw, 0)
      h_bw = tf.expand_dims(state_bw, 0)
      h = tf.concat((h_fw, h_bw), axis=0)
      output_fw = tf.expand_dims(output_fw, 1)
      output_bw = tf.expand_dims(output_bw, 1)
      output = tf.concat((output_fw, output_bw), axis=1)

    return [output, h] if output_sequence == 0 else [h]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_3(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls._common(node, **kwargs)
