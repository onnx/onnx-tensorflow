import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.handlers.handler import tf_op
from .math_mixin import ArithmeticMixin


@onnx_op("Sum")
@tf_op("AddN")
@tf_func(tf.add_n)
class Add(ArithmeticMixin, BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    tensor_dict = kwargs.pop("tensor_dict", {})
    return [cls.make_tf_node(
        node,
        inputs=[[tensor_dict.get(inp, None) for inp in node.inputs]],
        **kwargs)]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls._common(node, **kwargs)
