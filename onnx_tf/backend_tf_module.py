import tensorflow as tf
from onnx_tf.pb_wrapper import OnnxNode


class TFModuleHelper(object):
  """ Helper class for BackendTFModule and TFModule
  """

  # create tf.Variable for handlers that required to use variable in handler
  @classmethod
  def _create_handlers_variables_for_graph(cls,
                                           handlers,
                                           graph,
                                           init_dict,
                                           var_dict=None):
    var_dict = dict() if var_dict is None else var_dict
    for node in graph.node:
      var_dict = cls._create_handler_variables_for_node(handlers,
                                                        OnnxNode(node),
                                                        init_dict, var_dict)
    return var_dict

  @classmethod
  def _create_handler_variables_for_node(cls,
                                         handlers,
                                         node,
                                         init_dict=None,
                                         var_dict=None):
    init_dict = dict() if init_dict is None else init_dict
    var_dict = dict() if var_dict is None else var_dict
    handler = handlers[node.domain].get(
        node.op_type, None) if node.domain in handlers else None
    var_dict = handler.create_variables(
        handlers, node, init_dict, var_dict,
        cls._create_handlers_variables_for_graph) if handler else var_dict
    return var_dict


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
        graph_def)
    self.handler_variables = TFModuleHelper._create_handlers_variables_for_graph(
        handlers, graph_def, self.initializer_dict)
    self.is_export = False

  # get initializer from the main graph and all subgraphs in loop or if or scan
  # into tensor_dict
  def _get_initializer_from_graph_and_subgraphs(self, graph, init_dict=None):
    init_dict = dict() if init_dict is None else init_dict
    if graph.initializer:
      init_dict.update(
          self.backend._onnx_initializer_to_input_dict_items(graph.initializer))
    for node in graph.node:
      handler = self.handlers[node.domain].get(
          node.op_type, None) if node.domain in self.handlers else None
      init_dict = handler.get_initializer_from_subgraph(
          OnnxNode(node), init_dict, self.
          _get_initializer_from_graph_and_subgraphs) if handler else init_dict
    return init_dict

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

    outputs = dict()
    for output in self.outputs:
      if not self.is_export or tensor_dict[output].shape.is_fully_defined():
        outputs[output] = tensor_dict[output]
      else:
        # Restore the output shape if not fully defined during export
        for o in self.graph_def.output:
          if o.name == output:
            o_shape = [d.dim_value for d in o.type.tensor_type.shape.dim]
            outputs[
                output] = tensor_dict[output] if 0 in o_shape else tf.reshape(
                    tensor_dict[output], o_shape)
            break

    return outputs


class TFModule(tf.Module):
  """ TFModule is the tf.Module class used in backend.run_node.
  """

  def __init__(self, node, backend):
    super(TFModule, self).__init__()
    self.node = node
    self.backend = backend
    self.handlers = backend._get_handlers(opset=None)
    self.handler_variables = TFModuleHelper._create_handler_variables_for_node(
        self.handlers, node)

  @tf.function
  def __call__(self, **input_dict):
    input_dict.update(self.handler_variables)
    outputs = self.backend._onnx_node_to_tensorflow_op(self.node, input_dict,
                                                       self.handlers)
    return outputs
