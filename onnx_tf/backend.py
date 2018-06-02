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
from onnx_tf.common import attr_converter
from onnx_tf.common import attr_translator
from onnx_tf.common import data_type
from onnx_tf.common import exception
from onnx_tf.common import get_device_option
from onnx_tf.common import supports_device  # noqa
from onnx_tf.common.handler_helper import get_all_backend_handlers


# TODO: Move this into ONNX main library
class OnnxNode(object):
  """
  Reimplementation of NodeProto from ONNX, but in a form
  more convenient to work with from Python.
  """

  def __init__(self, node):
    self.name = str(node.name)
    self.op_type = str(node.op_type)
    self.domain = str(node.domain)
    self.attrs = dict([(attr.name,
                        attr_translator.translate_onnx(
                            attr.name, attr_converter.onnx2tf(attr)))
                       for attr in node.attribute])
    self.inputs = list(node.input)
    self.outputs = list(node.output)
    self.node_proto = node


class TensorflowBackend(Backend):
  """ Tensorflow Backend for ONNX
  """

  @classmethod
  def prepare(cls, model, device='CPU', **kwargs):
    """Prepare an ONNX model for Tensorflow Backend

    This function converts an ONNX model to an internel representation
    of the computational graph called TensorflowRep and returns
    the converted representation.

    :param model: the ONNX model to be converted
    :param device: the device to execute this model on

    :returns: a TensorflowRep class object representing the ONNX model
    """
    super(TensorflowBackend, cls).prepare(model, device, **kwargs)

    return cls.onnx_model_to_tensorflow_rep(model)

  @classmethod
  def onnx_model_to_tensorflow_rep(cls, model):
    return cls._onnx_graph_to_tensorflow_rep(model.graph, model.opset_import)

  @classmethod
  def _onnx_graph_to_tensorflow_rep(cls, graph_def, opset):
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
        x = tf.placeholder(
            data_type.onnx2tf(value_info.type.tensor_type.elem_type),
            name=value_info.name,
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
        output_ops = cls._onnx_node_to_tensorflow_op(
            onnx_node, tensor_dict, handlers, opset=opset)
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
      input_dict = dict(
          [(x[0], tf.constant(x[1])) for x in feed_dict_raw.items()])
      ops = cls._onnx_node_to_tensorflow_op(node, input_dict)

      with tf.Session() as sess:
        with tf.device(device_option):
          sess.run(tf.global_variables_initializer())
          output_vals = sess.run(ops)

    return namedtupledict('Outputs', node.outputs)(*output_vals)

  @classmethod
  def _onnx_initializer_to_input_dict_items(cls, initializer):

    def tensor2list(onnx_tensor):
      # Use the onnx.numpy_helper because the data may be raw
      return numpy_helper.to_array(onnx_tensor).flatten().tolist()

    return [(init.name,
             tf.constant(
                 tensor2list(init),
                 shape=init.dims,
                 dtype=data_type.onnx2tf(init.data_type)))
            for init in initializer]

  @classmethod
  def _onnx_node_to_tensorflow_op(cls,
                                  node,
                                  tensor_dict,
                                  handlers=None,
                                  opset=None):
    """
    Convert onnx node to tensorflow op.

    Args:
      node: Onnx node object.
      input_dict: Inputs dict of graph.
      opset: Opset version of the operator set. Default 0 means using latest version.

    Returns:
      Tensorflow op
    """
    handlers = handlers or cls._get_handlers(opset)
    handler = handlers[node.domain].get(node.op_type, None)
    if handler:
      return handler.handle(node, tensor_dict=tensor_dict)
    else:
      exception.OP_UNIMPLEMENTED_EXCEPT(node.op_type)

  @classmethod
  def _get_handlers(cls, opset):
    opset = opset or [make_opsetid(defs.ONNX_DOMAIN, defs.onnx_opset_version())]
    opset_dict = dict([(o.domain, o.version) for o in opset])
    return get_all_backend_handlers(opset_dict)


prepare = TensorflowBackend.prepare

run_node = TensorflowBackend.run_node

run_model = TensorflowBackend.run_model
