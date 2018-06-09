import copy

import numpy as np
import tensorflow as tf

from onnx_tf.common import exception
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Upsample")
@tf_func(tf.image.resize_images)
class Upsample(BackendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    x_shape = x.get_shape().as_list()
    if len(x_shape) != 4:
      exception.OP_UNSUPPORTED_EXCEPT("Upsample without 4D input", "Tensorflow")

    if node.attrs.get("mode", "nearest").lower() not in ["nearest", "bilinear"]:
      exception.OP_UNSUPPORTED_EXCEPT("Upsample without nearest or bilinear",
                                      "Tensorflow")

  @classmethod
  def version_7(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    x_shape = x.get_shape().as_list()
    attrs = copy.deepcopy(node.attrs)
    scales = attrs["scales"]
    new_height = np.floor(x_shape[2] * scales[2])
    new_weight = np.floor(x_shape[3] * scales[3])

    mode = attrs.get("mode", "nearest")
    if mode.lower() == "bilinear":
      mode = tf.image.ResizeMethod.BILINEAR
    else:
      mode = tf.image.ResizeMethod.NEAREST_NEIGHBOR

    attrs["size"] = np.array((new_height, new_weight), dtype=np.int32)
    attrs["method"] = mode

    return [
        cls.make_tensor_from_onnx_node(
            node, attrs=attrs, c_last_only=True, **kwargs)
    ]
