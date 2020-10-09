from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from onnx.backend.base import BackendRep, namedtupledict


class TensorflowRep(BackendRep):

  def __init__(self, graph=None, inputs=None, outputs=None, tensor_dict=None):
    super(TensorflowRep, self).__init__()
    self._graph = graph
    self._inputs = inputs or []
    self._outputs = outputs or []
    self._tensor_dict = tensor_dict or {}
    self._tf_module = None

  @property
  def graph(self):
    return self._graph

  @graph.setter
  def graph(self, graph):
    self._graph = graph

  @property
  def inputs(self):
    return self._inputs

  @inputs.setter
  def inputs(self, inputs):
    self._inputs = inputs

  @property
  def outputs(self):
    return self._outputs

  @outputs.setter
  def outputs(self, outputs):
    self._outputs = outputs

  @property
  def tensor_dict(self):
    return self._tensor_dict

  @tensor_dict.setter
  def tensor_dict(self, tensor_dict):
    self._tensor_dict = tensor_dict

  @property
  def tf_module(self):
    return self._tf_module

  @tf_module.setter
  def tf_module(self, tf_module):
    self._tf_module = tf_module

  def run(self, inputs, **kwargs):
    """ Run TensorflowRep.

    :param inputs: Given inputs.
    :param kwargs: Other args.
    :return: Outputs.
    """
    super(TensorflowRep, self).run(inputs, **kwargs)

    if isinstance(inputs, dict):
      feed_dict = inputs
    elif isinstance(inputs, list) or isinstance(inputs, tuple):
      if len(self.inputs) != len(inputs):
        raise RuntimeError('Expected {} values for uninitialized '
                           'graph inputs ({}), but got {}.'.format(
                               len(self.inputs), ', '.join(self.inputs),
                               len(inputs)))
      feed_dict = dict(zip(self.inputs, inputs))
    else:
      # single input
      feed_dict = dict([(self.inputs[0], inputs)])

    input_dict = dict(
        [(x[0], tf.constant(x[1])) for x in feed_dict.items()])

    output_values = self.tf_module(**input_dict)
    output_values = [val.numpy() if isinstance(val, tf.Tensor) else val for val in output_values]

    return namedtupledict('Outputs', self.outputs)(*output_values)

  def export_graph(self, path):
    """Export backend representation to a Tensorflow proto file.

    This function obtains the graph proto corresponding to the ONNX
    model associated with the backend representation and serializes
    to a protobuf file.

    :param path: The path to the output TF protobuf file.

    :returns: none.
    """
    tf.saved_model.save(self.tf_module, path, signatures=self.tf_module.__call__.get_concrete_function(**self.signatures))
