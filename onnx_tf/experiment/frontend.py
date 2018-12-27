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

    tf_graph = cls._make_tf_graph(graph_def, output, graph_name)
    parser = get_rnn_scope_parser(rnn_type)
    nodes = parser.parse(tf_graph.nodes)
    tf_graph.update_nodes(nodes)

    return cls._make_onnx_model(tf_graph, opset, producer_name,
                                ignore_unimplemented, optimizer_passes)
