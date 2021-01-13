import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Tile")
@tf_func(tf.tile)
class Tile(BackendHandler):
  cast_map = {tf.uint16: tf.uint32}

  @classmethod
  def _common(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    if cls.SINCE_VERSION < 6:  # opset 1
      x_rank = len(x.get_shape())
      tiles = kwargs["tensor_dict"][node.inputs[1]]
      axis = kwargs["tensor_dict"][node.inputs[2]]
      multiples = [1] * x_rank
      multiples[axis] = tiles
      return [
          cls.make_tensor_from_onnx_node(node, inputs=[x, multiples], **kwargs)
      ]
    else:  # opset 6 & 13
      x_dtype = x.dtype
      x = tf.cast(x, cls.cast_map[x_dtype]) if x_dtype in cls.cast_map else x
      repeats = kwargs["tensor_dict"][node.inputs[1]]
      output = tf.tile(x, repeats)
      output = tf.cast(output, x_dtype) if x_dtype in cls.cast_map else output
      return [output]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
