"""Backend for running ONNX on Tensorflow

To run this, you will need to have Tensorflow installed as well.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections

import numpy as np
from onnx import checker
from onnx.onnx_pb2 import GraphProto, TensorProto, AttributeProto
import onnx.numpy_helper
import onnx.defs
from onnx.backend.base import (
    Backend,
    BackendRep,
    Device,
    DeviceType,
    namedtupledict,
)

from onnx import onnx_pb2, helper
import tensorflow as tf


def get_device_option(device):
  m = {DeviceType.CPU: '/cpu:0',
       DeviceType.CUDA: '/gpu:0'}
  return m[device.type]

def get_type(onnx_type):
  pass

class TensorflowBackend(Backend):
  """ Tensorflow Backend for ONNX
  """
  @classmethod
  def run_node(cls, node, inputs, device='CPU'):
    super(TensorflowBackend, cls).run_node(node, inputs, device)

    device_option = get_device_option(Device(device))
    input_tensors = []
    for i in inputs:
      input_tensors.append(tf.constant(i))

    if isinstance(inputs, dict):
      feed_dict_raw = inputs
    else:
      assert len(node.input) == len(inputs)
      feed_dict_raw = dict(zip(node.input, inputs))

    input_dict = dict(map(lambda x: (x[0], tf.constant(x[1])), \
                          feed_dict_raw.items()))
    ops = cls._onnx_node_to_tensorflow_op(node, input_dict)
    output_vals = []
    with tf.Session() as sess:
      with tf.device(device_option):
        output_vals = map(sess.run, ops)
    return namedtupledict('Outputs', node.output)(*output_vals)

  @classmethod
  def _onnx_node_to_tensorflow_op(cls, node, input_dict):
    method_to_call = getattr(cls, "handle_" + node.op_type.lower())
    return method_to_call(node, input_dict)

  @classmethod
  def handle_relu(cls, node, input_dict):
    output_name = node.output[0]
    input_name = node.input[0]
    return [tf.nn.relu(input_dict[input_name])]

run_node = TensorflowBackend.run_node