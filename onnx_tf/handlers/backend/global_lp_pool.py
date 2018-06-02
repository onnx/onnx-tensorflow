import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("GlobalLpPool")
class GlobalLpPool(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    p = node.attrs.get("p", 2.)
    dims = list(range(len(x.shape)))
    dim_window = dims[2:]
    if len(dim_window) > 1 and p == 2:
      p = "euclidean"
    return [tf.norm(x, ord=p, axis=dim_window, keepdims=True)]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_2(cls, node, **kwargs):
    return cls._common(node, **kwargs)
