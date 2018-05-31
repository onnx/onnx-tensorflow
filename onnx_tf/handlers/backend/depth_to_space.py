import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.handlers.handler import tf_op


@onnx_op("DepthToSpace")
@tf_op("DepthToSpace")
@tf_func(tf.depth_to_space)
class DepthToSpace(BackendHandler):

  @classmethod
  def process_attrs(cls, attrs):
    return cls._process_attrs(attrs, rename={"blocksize": "block_size"})

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.make_tf_tensor(node, nc_cuda_only=True, **kwargs)
