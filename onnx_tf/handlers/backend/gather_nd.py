import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("GatherND")
class ScatterElements(BackendHandler):

  @classmethod
  def version_11(cls, node, **kwargs):
    params = kwargs["tensor_dict"][node.inputs[0]]
    indices = kwargs["tensor_dict"][node.inputs[1]]

    return [tf.gather_nd(params, indices)]