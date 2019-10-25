from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op

from onnx_tf.handlers.backend.scatter_elements import ScatterElements


@onnx_op("Scatter")
class Scatter(BackendHandler):

  @classmethod
  def version_9(cls, node, **kwargs):
    return ScatterElements.version_11(node, **kwargs)
