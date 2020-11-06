from onnx.defs import ONNX_DOMAIN
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
    self.initializer_dict = self._get_initializer_from_graph_and_subgraphs(
        self.graph_def, dict())
    self.handler_variables = self._create_handlers_variables(
        self.graph_def, dict())

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

  # create tf.Variable for handlers that required to use variable in handler
  def _create_handlers_variables(self, graph, vars_dict):
    handlers = self.backend._get_handlers(self.opset)
    for node in graph.node:
      handler = handlers[node.domain].get(
          node.op_type, None) if node.domain in handlers else None
      if handler and bool(handler.get_req_vars_template()):
        for v_name, v_template in handler.get_req_vars_template().items():
          v_init, v_shape = v_template
          v_count = 0
          for var_name in vars_dict.keys():
            v_count = v_count + 1 if var_name.startswith(v_name) else v_count
          v_name = v_name + '_' + str(v_count)
          vars_dict[v_name] = tf.Variable(v_init,
                                          dtype=v_init.dtype,
                                          shape=v_shape,
                                          name=v_name)
      if node.op_type in ['Loop', 'Scan']:
        onnx_node = OnnxNode(node)
        body = onnx_node.attrs["body"]
        vars_dict = self._create_handlers_variables(body, vars_dict)
      elif node.op_type == 'If':
        onnx_node = OnnxNode(node)
        then_branch = onnx_node.attrs['then_branch']
        vars_dict = self._create_handlers_variables(then_branch, vars_dict)
        else_branch = onnx_node.attrs['else_branch']
        vars_dict = self._create_handlers_variables(else_branch, vars_dict)
    return vars_dict

  @tf.function
  def gen_tensor_dict(self, input_dict):
    tensor_dict = dict(input_dict)
    tensor_dict.update(self.initializer_dict)
    tensor_dict.update(self.handler_variables)

    for node in self.graph_def.node:
      onnx_node = OnnxNode(node)
      output_ops = self.backend._onnx_node_to_tensorflow_op(onnx_node,
                                                            tensor_dict,
                                                            self.handlers,
                                                            opset=self.opset,
                                                            strict=self.strict)
      curr_node_output_map = dict(zip(onnx_node.outputs, output_ops))
      tensor_dict.update(curr_node_output_map)

    # reset VAR_COUNT in handlers(currently all handlers are in ONNX_DOMAIN)
    # TODO update this when we support handlers in other domain
    for _, handler in self.handlers[ONNX_DOMAIN].items():
      handler.VAR_COUNT = 0

    return tensor_dict

  @tf.function
  def __call__(self, **kwargs):
    tensor_dict = kwargs
    tensor_dict.update(self.initializer_dict)
    tensor_dict.update(self.handler_variables)

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

    # reset VAR_COUNT in handlers(currently all handlers are in ONNX_DOMAIN)
    # TODO update this when we support handlers in other domain
    for _, handler in self.handlers[ONNX_DOMAIN].items():
      handler.VAR_COUNT = 0
    return outputs
