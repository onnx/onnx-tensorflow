import tensorflow as tf
from onnx_tf.common import exception
from onnx_tf.common import get_variable_name
from onnx_tf.pb_wrapper import OnnxNode


class BackendTFModule(tf.Module):
  """ BackendTFModule is the tf.Module class used in backend.prepare,
  tf_rep.export_graph and tf_rep.run
  """

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
    if self.handlers:
      handlers = self.backend._get_handlers(self.opset)
      for node in graph.node:
        handler = handlers[node.domain].get(
            node.op_type, None) if node.domain in handlers else None
        if handler and bool(
            handler.get_req_vars_template(node, self.initializer_dict)):
          for v_name, v_template in handler.get_req_vars_template(
              node, self.initializer_dict).items():
            v_init, v_shape = v_template
            v_name = get_variable_name(node, v_name)
            if v_name in vars_dict.keys():
              # found duplicated variable name due to non unique node name
              exception.NON_UNIQUE_NODE_NAME_EXCEPT()
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

    return outputs


class TFModule(tf.Module):
  """ TFModule is the tf.Module class used in backend.run_node.
  """

  def __init__(self, node, backend):
    super(TFModule, self).__init__()
    self.node = node
    self.backend = backend
    self.handlers = backend._get_handlers(opset=None)
    self.handler_variables = self._create_handlers_variables(dict())

  def _create_handlers_variables(self, vars_dict):
    if self.handlers:
      handler = self.handlers[self.node.domain].get(
          self.node.op_type,
          None) if self.node.domain in self.handlers else None
      if handler and bool(
          handler.get_req_vars_template(self.node, self.node.attrs)):
        for v_name, v_template in handler.get_req_vars_template(
            self.node, self.node.attrs).items():
          v_init, v_shape = v_template
          v_name = get_variable_name(self.node, v_name)
          vars_dict[v_name] = tf.Variable(v_init,
                                          dtype=v_init.dtype,
                                          shape=v_shape,
                                          name=v_name)
    return vars_dict

  @tf.function
  def __call__(self, **input_dict):
    input_dict.update(self.handler_variables)
    outputs = self.backend._onnx_node_to_tensorflow_op(self.node, input_dict,
                                                       self.handlers)
    return outputs
