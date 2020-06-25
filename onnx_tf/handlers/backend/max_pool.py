from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import partial_support
from onnx_tf.handlers.handler import ps_description
from .pool_mixin import PoolMixin


@onnx_op("MaxPool")
@partial_support(True)
@ps_description(
    "MaxPoolWithArgmax with pad is None or incompatible mode, or " +
    "MaxPoolWithArgmax with 4D or higher input, or " +
    "MaxPoolWithArgmax with column major are not supported in Tensorflow.")
class MaxPool(PoolMixin, BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    pool_type = "MAX" if len(node.outputs) == 1 else "MAX_WITH_ARGMAX"
    return cls.pool(node, kwargs["tensor_dict"], pool_type,
                    kwargs.get("strict", True))

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_8(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_10(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_12(cls, node, **kwargs):
    return cls._common(node, **kwargs)
