from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from onnx.backend.base import BackendRep, namedtupledict


class TensorflowRep(BackendRep):

  def __init__(self, predict_net):
    super(TensorflowRep, self).__init__()
    self.predict_net = predict_net

  def run(self, inputs, **kwargs):
    super(TensorflowRep, self).run(inputs, **kwargs)

    # TODO: handle name scope if necessary
    with self.predict_net.graph.as_default():
      with tf.Session() as sess:
        if isinstance(inputs, dict):
          feed_dict = inputs
        elif isinstance(inputs, list) or isinstance(inputs, tuple):
          if len(self.predict_net.external_input) != len(inputs):
            raise RuntimeError('Expected {} values for uninitialized '
                               'graph inputs ({}), but got {}.'.format(
                                   len(self.predict_net.external_input),
                                   ', '.join(self.predict_net.external_input),
                                   len(inputs)))
          feed_dict = dict(zip(self.predict_net.external_input, inputs))
        else:
          # single input
          feed_dict = dict([(self.predict_net.external_input[0], inputs)])

        feed_dict = {
            self.predict_net.tensor_dict[key]: feed_dict[key]
            for key in self.predict_net.external_input
        }

        sess.run(tf.global_variables_initializer())
        external_output = [
            self.predict_net.tensor_dict[output]
            for output in self.predict_net.external_output
        ]

        output_values = sess.run(external_output, feed_dict=feed_dict)
        return namedtupledict('Outputs',
                              self.predict_net.external_output)(*output_values)

  def export_graph(self, path):
    """Export backend representation to a Tensorflow proto file.

    This function obtains the graph proto corresponding to the ONNX
    model associated with the backend representation and serializes
    to a protobuf file.

    :param path: the path to the output TF protobuf file.

    :returns: none.
    """
    graph_proto = self.predict_net.graph.as_graph_def()
    file = open(path, "wb")
    file.write(graph_proto.SerializeToString())
    file.close()
