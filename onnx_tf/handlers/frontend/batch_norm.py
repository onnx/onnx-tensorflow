from onnx_tf.handlers.frontend_handler import FrontendHandler


class BatchNorm(FrontendHandler):
  _TF_OP = ["BatchNorm", "FusedBatchNorm"]
  _ONNX_OP = "BatchNormalization"

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.make_node(
        node,
        version=1,
        epsilon=node.attr.get("epsilon", 1e-5),
        is_test=node.attr.get("is_training", 0),
        consumed_inputs=node.attr.get("consumed_inputs", [0, 0, 0, 1, 1]))

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.make_node(
        node,
        version=6,
        epsilon=node.attr.get("epsilon", 1e-5),
        is_test=node.attr.get("is_training", 0))
