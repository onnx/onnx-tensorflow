"""Backend for running ONNX on Tensorflow

To run this, you will need to have Tensorflow installed as well.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

from onnx_tf.backend import TensorflowBackendBase


class TensorflowBackend(TensorflowBackendBase):
  """ Tensorflow Backend for ONNX
  """

  @classmethod
  def handle_global_lp_pool(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    p = node.attrs.get("p", 2)
    dims = list(range(len(x.shape)))
    dim_window = dims[2:]
    if len(dim_window) > 1 and p == 2:
      p = "euclidean"
    return [tf.norm(x, ord=p, axis=dim_window, keepdims=True)]

  @classmethod
  def handle_pad(cls, node, input_dict):
    num_dim = int(len(node.attrs["pads"]) / 2)
    mode = node.attrs["mode"]

    def _compatibility_edge_pad(x, pads):
      x = np.pad(x, pads, mode="edge")
      return x

    value = node.attrs.get("value", 0)
    # tf requires int32 paddings
    pads = tf.constant(
        np.transpose(
            np.array(node.attrs["pads"]).reshape([2, num_dim])
            .astype(np.int32)))

    x = input_dict[node.inputs[0]]
    if mode.lower() == "edge":
      return [tf.py_func(_compatibility_edge_pad, [x, pads], x.dtype)]

    return [tf.pad(input_dict[node.inputs[0]], pads, mode, None, value)]

  @classmethod
  def handle_split(cls, node, input_dict):
    x_shape = input_dict[node.inputs[0]].get_shape().as_list()
    axis = node.attrs.get("axis", 0)
    if "split" in node.attrs:
      split = node.attrs["split"]
    else:
      per_part = x_shape[axis] / len(node.outputs)
      assert int(per_part) == per_part
      split = [int(per_part)] * len(node.outputs)
    return list(tf.split(input_dict[node.inputs[0]], split, axis))
