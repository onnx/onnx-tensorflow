from tensorflow.core.framework import graph_pb2

from onnx_tf.frontend import tensorflow_graph_to_onnx_model
from onnx_tf.pb_wrapper import TensorflowGraph

graph_def = graph_pb2.GraphDef()
with open("input_path", "rb") as f:  # load tf graph def
  graph_def.ParseFromString(f.read())
output = TensorflowGraph.get_output_node_names(
    graph_def)  # get output node names

model = tensorflow_graph_to_onnx_model(graph_def,
                                       output)  # convert tf graph to onnx model
with open("output_path", 'wb') as f:
  f.write(model.SerializeToString())
