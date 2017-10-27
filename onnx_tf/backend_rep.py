from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx.backend.base import BackendRep, namedtupledict
import tensorflow as tf

class TensorflowRep(BackendRep):
  def __init__(self, predict_net, input_dict, uninitialized):
    super(TensorflowRep, self).__init__()
    self.predict_net = predict_net
    self.input_dict = input_dict
    # The list of uninitialized external_inputs in workspace, we need this to
    # pair the name with given sequence inputs.
    self.uninitialized = uninitialized

  def run(self, inputs, **kwargs):
    super(TensorflowRep, self).run(inputs, **kwargs)
    # TODO: handle name scope if necessary
    with tf.Session() as sess:
      if isinstance(inputs, dict):
        feed_dict = inputs
      elif isinstance(inputs, list) or isinstance(inputs, tuple):
        if len(self.uninitialized) != len(inputs):
          raise RuntimeError('Expected {} values for uninitialized '
                     'graph inputs ({}), but got {}.'.format(
                       len(self.uninitialized),
                       ', '.join(self.uninitialized),
                       len(inputs)))
        feed_dict = dict(zip(self.uninitialized, inputs))
      else:
        # single input
        input_dict = dict([(self.uninitialized[0], inputs)])
      feed_dict = { self.input_dict[key]: feed_dict[key] for key in self.uninitialized }
      output_values = sess.run(self.predict_net.output_dict.values(), feed_dict=feed_dict)
      return namedtupledict('Outputs',
        self.predict_net.output_dict.keys())(*output_values)