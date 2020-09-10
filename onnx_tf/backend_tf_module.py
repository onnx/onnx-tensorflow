import tensorflow as tf
from onnx_tf.pb_wrapper import OnnxNode

class BackendTFModule(tf.Module):

  def __init__(self, handlers, opset, strict, graph_def, backend):
    super(BackendTFModule, self).__init__()
    self.handlers = handlers
    self.opset = opset
    self.strict = strict
    self.graph_def = graph_def
    self.backend = backend
    self.outputs = []

  @tf.function
  def __call__(self, **kwargs):
    tensor_dict = kwargs

    if self.graph_def.initializer:
      input_dict_items = self.backend._onnx_initializer_to_input_dict_items(
          self.graph_def.initializer)
    else:
      input_dict_items = []

    tensor_dict.update(input_dict_items)

    for node in self.graph_def.node:
      onnx_node = OnnxNode(node)
      output_ops = self.backend._onnx_node_to_tensorflow_op(
          onnx_node, tensor_dict, self.handlers, opset=self.opset, strict=self.strict)
      curr_node_output_map = dict(zip(onnx_node.outputs, output_ops))
      tensor_dict.update(curr_node_output_map)

    outputs = [tensor_dict[output] for output in self.outputs]
    return outputs
