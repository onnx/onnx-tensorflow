import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.handlers.handler import partial_support
from onnx_tf.handlers.handler import ps_description


@onnx_op("CastLike")
@tf_func(tf.cast)
@partial_support(True)
@ps_description("CastLike string to data types other than " +
                "float32/float64/int32/int64 is not supported in Tensorflow")
class CastLike(BackendHandler):

  @classmethod
  def get_attrs_processor_param(cls):
    return {"rename": {"to": "dtype"}}

  @classmethod
  def _common(cls, node, **kwargs):
    inp = kwargs["tensor_dict"][node.inputs[0]]
    like = kwargs["tensor_dict"][node.inputs[1]]
    to_type = like.dtype

    if to_type == tf.string:
      return [tf.as_string(inp)]

    if inp.dtype == tf.string:
      if to_type not in [tf.float32, tf.float64, tf.int32, tf.int64]:
        raise RuntimeError(
            "Cast string to type {} is not supported in Tensorflow.".format(
                to_type))
      return [tf.strings.to_number(inp, to_type)]

    return [
        cls.make_tensor_from_onnx_node(node, inputs=[inp, to_type], **kwargs)
    ]

  @classmethod
  def version_15(cls, node, **kwargs):
    return cls._common(node, **kwargs)
