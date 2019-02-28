from functools import partial

import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from .pool_mixin import PoolMixin


@onnx_op("MaxPool")
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
