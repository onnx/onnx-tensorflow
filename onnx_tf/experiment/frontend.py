from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx_tf.experiment.scope_parser import get_rnn_scope_parser
from onnx_tf.frontend import TensorflowFrontend


class ExperimentTensorflowFrontend(TensorflowFrontend):

  @classmethod
  def rnn_tf_graph_to_onnx_model(cls,
                                 graph_def,
                                 output,
                                 rnn_type,
                                 opset=0,
                                 producer_name="onnx-tensorflow",
                                 graph_name="graph",
                                 ignore_unimplemented=False,
                                 optimizer_passes=None):
    """EXPERIMENTAL
    Converts a RNN Tensorflow Graph Proto to an ONNX model

    This function converts a Tensorflow Graph proto to an equivalent
    representation of ONNX model.

    DO NOT DEFINE customized scope name in tf.dynamic_rnn and RNN cell.

    :param graph_def: Tensorflow Graph Proto object.
    :param output: List of string or a string specifying the name
      of the output graph node.
    :param opset: Opset version number, list or tuple.
      Default is 0 means using latest version with domain ''.
      List or tuple items should be (str domain, int version number).
    :param rnn_type: The rnn type contained in graph, should be one of GRU, LSTM, RNN.
    :param producer_name: The name of the producer.
    :param graph_name: The name of the output ONNX Graph.
    :param ignore_unimplemented: Convert to ONNX model and ignore all the operators
      that are not currently supported by onnx-tensorflow.
      This is an experimental feature. By enabling this feature,
      the model would not be guaranteed to match the ONNX specifications.
    :param optimizer_passes: List of optimization names c.f.
      https://github.com/onnx/onnx/blob/master/onnx/optimizer.py for available
      optimization passes.

    :returns: The equivalent ONNX Model Proto object.
    """

    tf_graph = cls._make_tf_graph(graph_def, output, graph_name)
    parser = get_rnn_scope_parser(rnn_type)
    nodes = parser.parse(tf_graph.nodes)
    tf_graph.update_nodes(nodes)

    return cls._make_onnx_model(tf_graph, opset, producer_name,
                                ignore_unimplemented, optimizer_passes)


rnn_tf_graph_to_onnx_model = ExperimentTensorflowFrontend.rnn_tf_graph_to_onnx_model
