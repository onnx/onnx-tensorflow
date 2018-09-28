import copy

import numpy as np
import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("ConstantLike")
@tf_func(tf.constant)
class ConstantLike(BackendHandler):

  @classmethod
  def get_attrs_processor_param(cls):
    return {"default": {"dtype": 1, "value": 0.}}

  @classmethod
  def version_9(cls, node, **kwargs):
    attrs = copy.deepcopy(node.attrs)
    if node.inputs:
      inp = kwargs["tensor_dict"][node.inputs[0]]
      attrs["shape"] = inp.get_shape()
      attrs["dtype"] = attrs.get("dtype", inp.dtype)
      attrs["value"] = np.asarray(attrs.get("value", 0)).astype(
          attrs["dtype"].as_numpy_dtype)
    return [
        cls.make_tensor_from_onnx_node(node, inputs=[], attrs=attrs, **kwargs)
    ]
