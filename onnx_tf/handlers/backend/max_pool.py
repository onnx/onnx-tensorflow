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
                    partial(tf.nn.pool, pooling_type='MAX'), 'MAX')
