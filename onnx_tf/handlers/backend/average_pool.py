from functools import partial

import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from .pool_mixin import PoolMixin


@onnx_op("AveragePool")
class AveragePool(PoolMixin, BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    # spatial_dim = list(tensor_dict[node.inputs[0]].get_shape()[2:])
    # kernel_shape = node.attrs.get("kernel_shape", [])
    # global_pool = True
    # for i in range(len(spatial_dim)):
    #   global_pool = global_pool and (spatial_dim[i] < kernel_shape[i])
    #
    # if global_pool:
    #   return cls.handle_global_average_pool(node, tensor_dict)

    # 0 = cannot pad zero
    return cls.pool(node, kwargs["tensor_dict"],
                    partial(tf.nn.pool, pooling_type='AVG'), 'AVG')

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls.pool(node, kwargs["tensor_dict"],
                    partial(tf.nn.pool, pooling_type='AVG'), 'AVG')
