from functools import partial

import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from .pool_mixin import PoolMixin


@onnx_op("AveragePool")
class AveragePool(PoolMixin, BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.pool(node, kwargs["tensor_dict"],
                    partial(tf.nn.pool, pooling_type='AVG'), 'AVG',
                    kwargs.get("strict", True))

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls.pool(node, kwargs["tensor_dict"],
                    partial(tf.nn.pool, pooling_type='AVG'), 'AVG',
                    kwargs.get("strict", True))

  @classmethod
  def version_10(cls, node, **kwargs):
    return cls.pool_v11(node, kwargs["tensor_dict"], "AVG",
                    kwargs.get("strict", True))

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls.pool_v11(node, kwargs["tensor_dict"], "AVG",
                    kwargs.get("strict", True))
