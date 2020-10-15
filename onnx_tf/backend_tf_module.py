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

  # get initializer from the main graph and all subgraphs in loop or if or scan
  # into tensor_dict
  def _get_initializer_from_graph_and_subgraphs(self, graph, graph_tensor_dict):
    if graph.initializer:
      graph_tensor_dict.update(
          self.backend._onnx_initializer_to_input_dict_items(graph.initializer))
    for node in graph.node:
      if node.op_type in ['Loop', 'Scan']:
        onnx_node = OnnxNode(node)
        body = onnx_node.attrs["body"]
        graph_tensor_dict = self._get_initializer_from_graph_and_subgraphs(
            body, graph_tensor_dict)
      elif node.op_type == 'If':
        onnx_node = OnnxNode(node)
        then_branch = onnx_node.attrs['then_branch']
        graph_tensor_dict = self._get_initializer_from_graph_and_subgraphs(
            then_branch, graph_tensor_dict)
        else_branch = onnx_node.attrs['else_branch']
        graph_tensor_dict = self._get_initializer_from_graph_and_subgraphs(
            else_branch, graph_tensor_dict)
    return graph_tensor_dict

  @tf.function
  def gen_tensor_dict(self, input_dict):
    tensor_dict = self._get_initializer_from_graph_and_subgraphs(
        self.graph_def, dict(input_dict))

    for node in self.graph_def.node:
      onnx_node = OnnxNode(node)
      output_ops = self.backend._onnx_node_to_tensorflow_op(onnx_node,
                                                            tensor_dict,
                                                            self.handlers,
                                                            opset=self.opset,
                                                            strict=self.strict)
      curr_node_output_map = dict(zip(onnx_node.outputs, output_ops))
      tensor_dict.update(curr_node_output_map)

    return tensor_dict

  @tf.function
  def __call__(self, **kwargs):
    tensor_dict = self._get_initializer_from_graph_and_subgraphs(
        self.graph_def, kwargs)

    for node in self.graph_def.node:
      onnx_node = OnnxNode(node)
      output_ops = self.backend._onnx_node_to_tensorflow_op(onnx_node,
                                                            tensor_dict,
                                                            self.handlers,
                                                            opset=self.opset,
                                                            strict=self.strict)
      curr_node_output_map = dict(zip(onnx_node.outputs, output_ops))
      tensor_dict.update(curr_node_output_map)

    outputs = [tensor_dict[output] for output in self.outputs]
    return outputs
