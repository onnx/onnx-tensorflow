import numpy as np

from onnx import mapping
from onnx import TensorProto
import tensorflow as tf

from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("Cast")
@tf_op("Cast")
class Cast(FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    dst_t = TensorProto.DataType.Name(mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(
        tf.as_dtype(node.attr["DstT"]).as_numpy_dtype)])
    return cls.make_node_from_tf_node(node, [node.inputs[0]], to=dst_t)

  @classmethod
  def version_6(cls, node, **kwargs):
    dst_t = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(
        tf.as_dtype(node.attr["DstT"]).as_numpy_dtype)]
    return cls.make_node_from_tf_node(node, [node.inputs[0]], to=dst_t)
