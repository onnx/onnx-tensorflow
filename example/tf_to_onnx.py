from tensorflow.core.framework import graph_pb2

from onnx_tf.common import get_output_node_names
from onnx_tf.frontend import tensorflow_graph_to_onnx_model


graph_def = graph_pb2.GraphDef()
with open("input_path", "rb") as f:
  graph_def.ParseFromString(f.read())
output = get_output_node_names(graph_def)

model = tensorflow_graph_to_onnx_model(graph_def, output, ignore_unimplemented=True)
with open("output_path", 'wb') as f:
  f.write(model.SerializeToString())
