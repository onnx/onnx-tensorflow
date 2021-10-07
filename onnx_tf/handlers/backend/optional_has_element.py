from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("OptionalHasElement")
class OptionalHasElement(BackendHandler):

  @classmethod
  def version_15(cls, node, **kwargs):
    if len(node.inputs) > 0 and kwargs["tensor_dict"][node.inputs[0]] is not None:
        return [True]
    else:
        return [False]
