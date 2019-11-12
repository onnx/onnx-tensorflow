from functools import partial

import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import partial_support
from onnx_tf.handlers.handler import ps_description
from .pool_mixin import PoolMixin


@onnx_op("MaxPool")
@partial_support(True)
@ps_description("MaxPoolWithArgmax with pad is None or incompatible mode, or " +
                "MaxPoolWithArgmax with 4D or higher input, or" +
                "MaxPoolWithArgmax with column major " +
                "are not supported in Tensorflow.")
class MaxPool(PoolMixin, BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.pool(node, kwargs["tensor_dict"],
                    partial(tf.nn.pool, pooling_type='MAX'), 'MAX',
                    kwargs.get("strict", True))

  @classmethod
  def version_8(cls, node, **kwargs):
    if len(node.outputs) == 1:
      pool_type = "MAX"
      pool_func = partial(tf.nn.pool, pooling_type='MAX')
    else:
      pool_type = 'MAX_WITH_ARGMAX'
      pool_func = tf.nn.max_pool_with_argmax
    return cls.pool(node, kwargs["tensor_dict"], pool_func, pool_type,
                    kwargs.get("strict", True))

  @classmethod
  def version_10(cls, node, **kwargs):
    pool_type = "MAX" if len(node.outputs) == 1 else "MAX_WITH_ARGMAX"
    return cls.pool_v11(node, kwargs["tensor_dict"], pool_type,
                        kwargs.get("strict", True))

  @classmethod
  def version_11(cls, node, **kwargs):
    pool_type = "MAX" if len(node.outputs) == 1 else "MAX_WITH_ARGMAX"
    return cls.pool_v11(node, kwargs["tensor_dict"], pool_type,
                        kwargs.get("strict", True))
