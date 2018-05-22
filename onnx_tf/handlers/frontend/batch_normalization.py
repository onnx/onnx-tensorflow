from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.frontend_handler import version


class BatchNormalization(FrontendHandler):
  TF_OP = ["BatchNorm", "FusedBatchNorm"]

  @classmethod
  @version(1)
  def version_1(cls, node, **kwargs):
    return cls.make_node(
        node,
        epsilon=node.attr.get("epsilon", 1e-5),
        is_test=node.attr.get("is_training", 0),
        consumed_inputs=node.attr.get("consumed_inputs", [0, 0, 0, 1, 1]))

  @classmethod
  @version(6)
  def version_6(cls, node, **kwargs):
    return cls.make_node(
        node,
        epsilon=node.attr.get("epsilon", 1e-5),
        is_test=node.attr.get("is_training", 0))
