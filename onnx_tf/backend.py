"""Backend for running ONNX on Tensorflow

To run this, you will need to have Tensorflow installed as well.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

try:
  from itertools import izip as zip
except ImportError:  # will be 3.x series
  pass

from onnx import defs
from onnx import numpy_helper
from onnx.backend.base import Backend
from onnx.backend.base import Device
from onnx.backend.base import namedtupledict
from onnx.helper import make_opsetid
import tensorflow as tf

from onnx_tf.backend_rep import TensorflowRep
from onnx_tf.common import data_type
from onnx_tf.common import exception
from onnx_tf.common import get_device_option
from onnx_tf.common import get_unique_suffix
from onnx_tf.common import supports_device as common_supports_device
from onnx_tf.common.handler_helper import get_all_backend_handlers
from onnx_tf.pb_wrapper import OnnxNode
import onnx_tf.common as common


class TensorflowBackend(Backend):
  """ Tensorflow Backend for ONNX
  """

  @classmethod
  def prepare(cls,
              model,
              device='CPU',
              strict=True,
              logging_level='INFO',
              **kwargs):
    """Prepare an ONNX model for Tensorflow Backend.

    This function converts an ONNX model to an internel representation
    of the computational graph called TensorflowRep and returns
    the converted representation.

    :param model: The ONNX model to be converted.
    :param device: The device to execute this model on.
    :param strict: Whether to enforce semantic equivalence between the original model
      and the converted tensorflow model, defaults to True (yes, enforce semantic equivalence).
      Changing to False is strongly discouraged.
      Currently, the strict flag only affects the behavior of MaxPool and AveragePool ops.
    :param logging_level: The logging level, default is INFO. Change it to DEBUG
      to see more conversion details or to WARNING to see less

    :returns: A TensorflowRep class object representing the ONNX model
    """
    super(TensorflowBackend, cls).prepare(model, device, **kwargs)
    common.logger.setLevel(logging_level)
    common.logger.handlers[0].setLevel(logging_level)

    return cls.onnx_model_to_tensorflow_rep(model, strict)

  @classmethod
  def onnx_model_to_tensorflow_rep(cls, model, strict):
    """ Convert ONNX model to TensorflowRep.

    :param model: ONNX ModelProto object.
    :param strict: whether to enforce semantic equivalence between the original model
      and the converted tensorflow model.
    :return: TensorflowRep object.
    """

    # Models with IR_VERSION less than 3 does not have opset_import set.
    # We default to minimum opset, this behavior is consistent with
    # onnx checker.
    # c.f. https://github.com/onnx/onnx/blob/427ac0c1b792363d373e3d7e4eef97fa46458420/onnx/checker.cc#L478
    if model.ir_version < 3:
      opset_import = [make_opsetid(defs.ONNX_DOMAIN, 1)]
    else:
      opset_import = model.opset_import
    return cls._onnx_graph_to_tensorflow_rep(model.graph, opset_import, strict)

  @classmethod
  def _onnx_graph_to_tensorflow_rep(cls, graph_def, opset, strict):
    """ Convert ONNX graph to TensorflowRep.

    :param graph_def: ONNX GraphProto object.
    :param opset: ONNX OperatorSetIdProto list.
    :param strict: whether to enforce semantic equivalence between the original model
      and the converted tensorflow model.
    :return: TensorflowRep object.
    """
    handlers = cls._get_handlers(opset)

    tf_rep_graph = tf.Graph()
    with tf_rep_graph.as_default():
      # initializer: TensorProtos representing the values to initialize
      # a given tensor.
      # initialized: A list of names of the initialized tensors.
      if graph_def.initializer:
        input_dict_items = cls._onnx_initializer_to_input_dict_items(
            graph_def.initializer)
        initialized = {init.name for init in graph_def.initializer}
      else:
        input_dict_items = []
        initialized = set()

      # creating placeholders for currently unknown inputs
      for value_info in graph_def.input:
        if value_info.name in initialized:
          continue
        shape = list(
            d.dim_value if (d.dim_value > 0 and d.dim_param == "") else None
            for d in value_info.type.tensor_type.shape.dim)
        value_info_name = value_info.name.replace(
            ":", "_tf_") + "_" + get_unique_suffix(
            ) if ":" in value_info.name else value_info.name

        x = tf.compat.v1.placeholder(data_type.onnx2tf(
            value_info.type.tensor_type.elem_type),
                                     name=value_info_name,
                                     shape=shape)
        input_dict_items.append((value_info.name, x))

      # tensor dict: this dictionary is a map from variable names
      # to the latest produced TF tensors of the given name.
      # This dictionary will get updated as we build the graph to
      # record the names of newly produced tensors.
      tensor_dict = dict(input_dict_items)
      # Since tensor dict may be updated, we need to keep a copy
      # of the original input dict where we track the earliest
      # defined tensors so we can have access to the placeholders
      # to feed in input tensors when we run the graph.
      input_dict = dict(input_dict_items)

      for node in graph_def.node:
        onnx_node = OnnxNode(node)
        output_ops = cls._onnx_node_to_tensorflow_op(onnx_node,
                                                     tensor_dict,
                                                     handlers,
                                                     opset=opset,
                                                     strict=strict)
        curr_node_output_map = dict(zip(onnx_node.outputs, output_ops))
        tensor_dict.update(curr_node_output_map)

    tf_rep = TensorflowRep()
    tf_rep.graph = tf_rep_graph
    tf_rep.inputs = [
        value_info.name
        for value_info in graph_def.input
        if value_info.name not in initialized
    ]
    tf_rep.outputs = [value_info.name for value_info in graph_def.output]
    tf_rep.tensor_dict = tensor_dict
    return tf_rep

  @classmethod
  def run_node(cls, node, inputs, device='CPU', outputs_info=None, **kwargs):
    """ Run ONNX node.

    :param node: ONNX NodeProto object.
    :param inputs: Inputs.
    :param device: Device run on.
    :param outputs_info: None.
    :param kwargs: Other args.
    :return: Outputs.
    """
    super(TensorflowBackend, cls).run_node(node, inputs, device)
    node_graph = tf.Graph()
    with node_graph.as_default():
      node = OnnxNode(node)
      device_option = get_device_option(Device(device))
      input_tensors = []
      for i in inputs:
        input_tensors.append(tf.constant(i))

      if isinstance(inputs, dict):
        feed_dict_raw = inputs
      else:
        assert len(node.inputs) == len(inputs)
        feed_dict_raw = dict(zip(node.inputs, inputs))

      # TODO: is constant the best way for feeding inputs?
      input_dict = dict([
          (x[0], tf.constant(x[1])) for x in feed_dict_raw.items()
      ])
      ops = cls._onnx_node_to_tensorflow_op(node, input_dict)

      with tf.compat.v1.Session() as sess:
        with tf.device(device_option):
          sess.run(tf.compat.v1.global_variables_initializer())
          output_vals = sess.run(ops)

    return namedtupledict('Outputs', node.outputs)(*output_vals)

  @classmethod
  def _onnx_initializer_to_input_dict_items(cls, initializer):
    """ Convert ONNX graph initializer to input dict items.

    :param initializer: ONNX graph initializer, list of TensorProto.
    :return: List of input dict items.
    """

    def tensor2list(onnx_tensor):
      # Use the onnx.numpy_helper because the data may be raw
      return numpy_helper.to_array(onnx_tensor).flatten().tolist()

    def validate_initializer_name(name):
      # Prepend a unique suffix if leading charater is "_"
      name = get_unique_suffix() + name if name[0] is "_" else name

      # Replace ":" with "_tf_" and append a unique suffix for
      # traceability
      return name.replace(
          ":", "_tf_") + "_" + get_unique_suffix() if ":" in name else name

    return [(init.name,
             tf.constant(tensor2list(init),
                         shape=init.dims,
                         dtype=data_type.onnx2tf(init.data_type),
                         name=validate_initializer_name(init.name)))
            for init in initializer]

  @classmethod
  def _onnx_node_to_tensorflow_op(cls,
                                  node,
                                  tensor_dict,
                                  handlers=None,
                                  opset=None,
                                  strict=True):
    """
    Convert onnx node to tensorflow op.

    Args:
      node: Onnx node object.
      tensor_dict: Tensor dict of graph.
      opset: Opset version of the operator set. Default 0 means using latest version.
      strict: whether to enforce semantic equivalence between the original model
        and the converted tensorflow model, defaults to True (yes, enforce semantic equivalence).
        Changing to False is strongly discouraged.
    Returns:
      Tensorflow op
    """
    handlers = handlers or cls._get_handlers(opset)
    handler = handlers[node.domain].get(node.op_type, None)
    if handler:
      return handler.handle(node, tensor_dict=tensor_dict, strict=strict)
    else:
      exception.OP_UNIMPLEMENTED_EXCEPT(node.op_type)

  @classmethod
  def _get_handlers(cls, opset):
    """ Get all backend handlers with opset.

    :param opset: ONNX OperatorSetIdProto list.
    :return: All backend handlers.
    """
    opset = opset or [make_opsetid(defs.ONNX_DOMAIN, defs.onnx_opset_version())]
    opset_dict = dict([(o.domain, o.version) for o in opset])
    return get_all_backend_handlers(opset_dict)

  @classmethod
  def supports_device(cls, device):
    return common_supports_device(device)

  @classmethod
  def onnx_graph_to_tensorflow_ops(cls,
                                   subgraph,
                                   input_values,
                                   tensor_dict,
                                   opset=None,
                                   strict=True):
    """
    Converts ONNX graph to Tensorflow operations
    Args:
      subgraph:         the ONNX graph to be converted
      input_values:     dictionary with values/tensors to initialize
                        the subgraph inputs. if the subgraph.input
                        are send in as parameters then it is required,
                        otherwise this can be empty dictionary
      tensor_dict:      the dictionary that contain values for all the
                        node.inputs in the subgraph that are not defined
                        in the subgraph or input_values.
      opset:            opset version of the operator set.
      strict:           whether to enforce semantic equivalence between the
                        original model and the converted tensorflow model,
                        defaults to True (yes, enforce semantic equivalence).
    Returns:
      array of Tensorflow Tensors
    """
    # get the subgraph.input from input_values
    subgraph_tensor_dict = input_values.copy()
    # get the rest of the subgraph input from tensor_dict
    for i in subgraph.input:
      if i.name not in subgraph_tensor_dict.keys():
        subgraph_tensor_dict[i.name] = tensor_dict[i.name]
    # get the required initializer constant node(s) for the subgraph
    # Need to get the initializer constant nodes from tensor_dict here
    # because input from initializer will not be send in as inputs
    # to the subgraph and those nodes are not in the subgraph
    nodes_outputs = []
    for node in subgraph.node:
      for o_name in node.output:
        nodes_outputs.append(o_name)
    for node in subgraph.node:
      for i_name in node.input:
        if i_name not in nodes_outputs and i_name not in subgraph_tensor_dict.keys():
          subgraph_tensor_dict[i_name] = tensor_dict[i_name]
      onnx_node = OnnxNode(node)
      output_ops = cls._onnx_node_to_tensorflow_op(onnx_node,
                                                   subgraph_tensor_dict,
                                                   opset=opset,
                                                   strict=strict)
      curr_node_output_map = dict(zip(onnx_node.outputs, output_ops))
      subgraph_tensor_dict.update(curr_node_output_map)
    return subgraph_tensor_dict

  @classmethod
  def onnx_graph_to_tensorflow_rep(cls, graph_def, strict=True):
    """
    Converts ONNX graph to TensorflowRep
    Args:
      graph_def:        the ONNX graph to be converted
      strict:           whether to enforce semantic equivalence between the
                        original model and the converted tensorflow model,
                        defaults to True (yes, enforce semantic equivalence).
    Returns:
      TensorflowRep object.
    """
    # get the opset of the installed ONNX
    opset = [make_opsetid(defs.ONNX_DOMAIN, defs.onnx_opset_version())]
    return cls._onnx_graph_to_tensorflow_rep(graph_def, opset, strict)


prepare = TensorflowBackend.prepare

run_node = TensorflowBackend.run_node

run_model = TensorflowBackend.run_model

supports_device = TensorflowBackend.supports_device

onnx_graph_to_tensorflow_ops = TensorflowBackend.onnx_graph_to_tensorflow_ops

onnx_graph_to_tensorflow_rep = TensorflowBackend.onnx_graph_to_tensorflow_rep
