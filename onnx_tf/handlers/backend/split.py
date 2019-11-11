import copy

import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Split")
@tf_func(tf.split)
class Split(BackendHandler):

  @classmethod
  def get_attrs_processor_param(cls):
    return {"default": {"axis": 0}}

  @classmethod
  def _common(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x_shape = tensor_dict[node.inputs[0]].get_shape().as_list()
    attrs = copy.deepcopy(node.attrs)
    axis = attrs.get("axis", 0)
    axis = axis if axis >= 0 else len(x_shape) + axis
    if "split" in node.attrs:
      split = attrs["split"]
    elif len(node.inputs) == 2:  # since version 1
      split = tensor_dict[node.inputs[1]]
    else:
      per_part = x_shape[axis] / len(node.outputs)
      if int(per_part) != per_part:
        raise ValueError("Split can not be evenly divided.")
      split = [int(per_part)] * len(node.outputs)
    attrs["num_or_size_splits"] = split
    return list(
        cls.make_tensor_from_onnx_node(
            node, inputs=[tensor_dict[node.inputs[0]]], attrs=attrs, **kwargs))

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_2(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)
