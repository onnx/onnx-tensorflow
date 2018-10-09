import onnx

from onnx_tf.backend import prepare


onnx_model = onnx.load("input_path")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("output_path")
