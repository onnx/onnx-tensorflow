import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Tile")
@tf_func(tf.tile)
class Tile(BackendHandler):

  @classmethod
  def get_attrs_processor_param(cls):
    return {"rename": {"axes": "axis"}}

  @classmethod
  def version_1(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    x_rank = len(x.get_shape())
    multiples = [1] * x_rank
    axis = node.attrs["axis"]
    tiles = node.attrs["tiles"]
    multiples[axis] = tiles
    inputs = [x, multiples]
    return [cls.make_tensor_from_onnx_node(node, inputs=inputs, **kwargs)]

  @classmethod
  def version_6(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]
