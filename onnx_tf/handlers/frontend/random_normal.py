from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.frontend_handler import version


class RandomNormal(FrontendHandler):
  TF_OP = ["RandomStandardNormal"]
  ONNX_OP = "RandomNormal"

  @classmethod
  @version(1)
  def version_1(cls, node, **kwargs):
    return cls.make_node(
        node, [],
        dtype=node.attr["dtype"],
        seed=node.attr["seed"],
        mean=0.0,
        scale=1.0,
        shape=node.attr["_output_shapes"][0])
