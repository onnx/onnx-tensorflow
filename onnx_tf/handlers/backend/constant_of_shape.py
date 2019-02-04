import copy

import tensorflow as tf

from onnx import numpy_helper

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("ConstantOfShape")
@tf_func(tf.fill)
class ConstantOfShape(BackendHandler):

  @classmethod
  def version_9(cls, node, **kwargs):
    attrs = copy.deepcopy(node.attrs)

    shape = kwargs["tensor_dict"][node.inputs[0]]

    # make sure the shape dtype is either int32 or int64
    if shape.dtype not in [tf.int64, tf.int32]:
      shape = tf.cast(shape, tf.int64)

    # the default value is 0, float32
    if "value" in node.attrs:
      attr_value = node.attrs["value"]
      value = numpy_helper.to_array(attr_value)
      attrs["value"] = value[0]
    else:
      attrs["value"] = 0.

    return [
        cls.make_tensor_from_onnx_node(
            node, inputs=[shape], attrs=attrs, **kwargs)
    ]
