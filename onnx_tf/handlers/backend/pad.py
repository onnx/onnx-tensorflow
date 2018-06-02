import copy

import numpy as np
import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Pad")
@tf_func(tf.pad)
class Pad(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    attrs = copy.deepcopy(node.attrs)
    pads = attrs.pop("pads", attrs.pop("paddings", None))
    num_dim = len(tensor_dict[node.inputs[0]].get_shape())
    mode = attrs.get("mode", "constant")
    attrs["constant_values"] = attrs.pop("value", 0.)

    def _compatibility_edge_pad(x, pads):
      x = np.pad(x, pads, mode="edge")
      return x

    # tf requires int32 paddings
    attrs["paddings"] = tf.constant(
        np.transpose(np.array(pads).reshape([2, num_dim]).astype(np.int32)))

    x = tensor_dict[node.inputs[0]]
    if mode.lower() == "edge":
      return [
          tf.py_func(_compatibility_edge_pad, [x, attrs["paddings"]], x.dtype)
      ]

    return [cls.make_tensor_from_onnx_node(node, attrs=attrs, **kwargs)]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_2(cls, node, **kwargs):
    return cls._common(node, **kwargs)
