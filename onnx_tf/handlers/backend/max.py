import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Max")
@tf_func(tf.reduce_max)
class Max(BackendHandler):

  @classmethod
  def process_attrs(cls, attrs):
    return cls._process_attrs(
        attrs, remove=["consumed_inputs"], default={"axis": 0})

  @classmethod
  def _common(cls, node, **kwargs):
    values = [kwargs["tensor_dict"][inp] for inp in node.inputs]
    return [cls.make_tf_tensor(node, inputs=[tf.stack(values)], **kwargs)]

  @classmethod
  def version_1(cls, node, **kwargs):
    return [cls._common(node, **kwargs)]

  @classmethod
  def version_6(cls, node, **kwargs):
    return [cls._common(node, **kwargs)]
