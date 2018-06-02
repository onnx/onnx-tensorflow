import numpy as np
import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.common import data_type


@onnx_op("Constant")
@tf_func(tf.constant)
class Constant(BackendHandler):

  @classmethod
  def process_attrs(cls, attrs):
    return cls._process_attrs(attrs, remove=["value"])

  @classmethod
  def version_1(cls, node, **kwargs):
    attr_value = node.attrs["value"]
    dtype = data_type.onnx2tf(attr_value.data_type)
    field = data_type.onnx2field(attr_value.data_type)
    value = np.reshape(getattr(attr_value, field), attr_value.dims)
    return [cls.make_tf_tensor(node, inputs=[value], attrs={"dtype": dtype})]
