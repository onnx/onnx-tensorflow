import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from .control_flow_mixin import ComparisonMixin


@onnx_op("Equal")
@tf_func(tf.equal)
class Equal(ComparisonMixin, BackendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    supported_dtype = [
        tf.bfloat16, tf.half, tf.float32, tf.float64, tf.uint8,
        tf.int8, tf.int16, tf.int32, tf.int64, tf.complex64,
        tf.quint8, tf.qint8, tf.qint32, tf.string, tf.bool, 
        tf.complex128
    ]
    x = kwargs["tensor_dict"][node.inputs[0]]
    if x.dtype not in supported_dtype:
      exception.OP_UNSUPPORTED_EXCEPT(
          "Equal input in " + str(x.dtype) + " which", "Tensorflow")

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.limited_broadcast(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]

  @classmethod
  def version_11(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]
