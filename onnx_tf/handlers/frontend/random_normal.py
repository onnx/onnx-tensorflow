from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("RandomNormal")
@tf_op("RandomStandardNormal")
class RandomNormal(FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.make_node_from_tf_node(
        node, [],
        dtype=node.attr["dtype"],
        seed=node.attr["seed"],
        mean=0.0,
        scale=1.0,
        shape=node.attr["_output_shapes"][0])
