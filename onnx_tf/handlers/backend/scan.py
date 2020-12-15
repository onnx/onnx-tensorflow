from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from .scan_mixin import ScanMixin


@onnx_op("Scan")
class Scan(ScanMixin, BackendHandler):

  @classmethod
  def get_initializer_from_subgraph(cls, node, init_dict, callback_func):
    return callback_func(node.attrs["body"], init_dict)

  @classmethod
  def create_variables(cls, handlers, node, init_dict, var_dict, callback_func):
    return callback_func(handlers, node.attrs["body"], init_dict, var_dict)

  @classmethod
  def _common(cls, node, **kwargs):
    return cls.scan(node, kwargs["tensor_dict"], kwargs.get("strict", True))

  @classmethod
  def version_8(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_9(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)
