from onnx_tf.handlers.frontend_handler import FrontendHandler


class RandomUniform(FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.make_node(
        node, [], [node.name],
        1,
        dtype=node.attr["dtype"],
        seed=node.attr["seed"],
        high=1.0,
        low=0.0,
        shape=node.attr["_output_shapes"][0])
