import numpy as np

from onnx import numpy_helper
import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.common import data_type


@onnx_op("Constant")
@tf_func(tf.constant)
class Constant(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    attr_value = node.attrs["value"]
    dtype = data_type.onnx2tf(attr_value.data_type)
    value = numpy_helper.to_array(attr_value)
    return [
        cls.make_tensor_from_onnx_node(node,
                                       inputs=[value],
                                       attrs={"dtype": dtype})
    ]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_9(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    # either value or sparse_value
    if "value" in node.attrs:
      return cls._common(node, **kwargs)
    else:
      sparse_value = node.attrs["sparse_value"]
      indices = numpy_helper.to_array(sparse_value.indices)
      values = numpy_helper.to_array(sparse_value.values)
      shape = np.array(sparse_value.dims)
    return [tf.SparseTensor(indices, values, shape)]

  @classmethod
  def version_12(cls, node, **kwargs):
    if "value" in node.attrs or "sparse_value" in node.attrs:
      return cls.version_11(node, **kwargs)
    elif "value_float" in node.attrs:
      value = node.attrs["value_float"]
      dtype = tf.float32
    elif "value_floats" in node.attrs:
      value = node.attrs["value_floats"]
      dtype = tf.float32
    elif "value_int" in node.attrs:
      value = node.attrs["value_int"]
      dtype = tf.int64
    elif "value_ints" in node.attrs:
      value = node.attrs["value_ints"]
      dtype = tf.int64
    elif "value_string" in node.attrs:
      value = node.attrs["value_string"]
      dtype = tf.string
    elif "value_strings" in node.attrs:
      value = node.attrs["value_strings"]
      dtype = tf.string
    return [
        cls.make_tensor_from_onnx_node(node,
                                       inputs=[value],
                                       attrs={"dtype": dtype})
    ]

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls.version_12(node, **kwargs)
