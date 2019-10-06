import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("ScatterND")
class ScatterND(BackendHandler):

  @classmethod
  def version_11(cls, node, **kwargs):
    data = kwargs["tensor_dict"][node.inputs[0]]
    indices = kwargs["tensor_dict"][node.inputs[1]]
    updates = kwargs["tensor_dict"][node.inputs[2]]

    return [tf.tensor_scatter_nd_update(data, indices, updates)]