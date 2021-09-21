import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.common.tf_helper import tf_shape


@onnx_op("Shape")
@tf_func(tf.shape)
class Shape(BackendHandler):

  @classmethod
  def get_attrs_processor_param(cls):
    return {"default": {"out_type": tf.int64}}

  @classmethod
  def _common(cls, node, **kwargs):
    if cls.SINCE_VERSION < 15:
      return [cls.make_tensor_from_onnx_node(node, **kwargs)]

    x = kwargs["tensor_dict"][node.inputs[0]]
    x_shape = tf_shape(x)
    x_rank = len(x_shape)

    start = node.attrs.get("start", 0)
    if start < 0:
      start += x_rank
      # Clip if start is still < 0
      start = 0 if start < 0 else start

    end = node.attrs.get("end", x_rank)
    if end < 0:
      end += x_rank
      # Clip if end is still < 0
      end = 0 if end < 0 else end


    result = cls.make_tensor_from_onnx_node(node, **kwargs)

    return [tf.slice(result, [start], [end - start])]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_15(cls, node, **kwargs):
    return cls._common(node, **kwargs) 
