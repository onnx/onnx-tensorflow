import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Cast")
@tf_func(tf.cast)
class Cast(BackendHandler):

  @classmethod
  def get_attrs_processor_param(cls):
    return {"rename": {"to": "dtype"}}

  @classmethod
  def version_1(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]

  @classmethod
  def version_6(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]

  @classmethod
  def version_9(cls, node, **kwargs):
    inp = kwargs["tensor_dict"][node.inputs[0]]
    to_type = node.attrs.get("to")

    if to_type == tf.string:
      return [tf.as_string(inp)]

    if inp.dtype == tf.string:
      if to_type not in [tf.float32, tf.float64, tf.int32, tf.int64]:
        raise RuntimeError(
            "Cast string to type {} is not supported in Tensorflow.".format(
                to_type))
      return [tf.strings.to_number(inp, to_type)]

    return [cls.make_tensor_from_onnx_node(node, **kwargs)]
