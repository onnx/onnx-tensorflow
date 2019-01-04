import onnx

from onnx_tf.backend import prepare

onnx_model = onnx.load("input_path")  # load onnx model
tf_rep = prepare(onnx_model)  # prepare tf representation
tf_rep.export_graph("output_path")  # export the model
