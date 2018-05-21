from onnx_tf.common.broadcast import get_broadcast_axis
from onnx_tf.handlers.frontend_handler import FrontendHandler


class LogicalCommon(FrontendHandler):

  @classmethod
  def logical_op(cls, node, broadcast=1, **kwargs):
    ex_kwargs = {}
    if broadcast == 1:
      ex_kwargs["broadcast"] = 1
      node_dict = kwargs["node_dict"]
      axis = get_broadcast_axis(*[node_dict[x] for x in node.inputs[0:2]])
      if axis is not None:
        ex_kwargs["axis"] = axis
    return cls.make_node(node, **ex_kwargs)


class ComparisonCommon(FrontendHandler):

  @classmethod
  def comparison_op(cls, node, broadcast=1, **kwargs):
    ex_kwargs = {}
    if broadcast == 1:
      ex_kwargs["broadcast"] = 1
      node_dict = kwargs["node_dict"]
      axis = get_broadcast_axis(*[node_dict[x] for x in node.inputs[0:2]])
      if axis is not None:
        ex_kwargs["axis"] = axis
    return cls.make_node(node, **ex_kwargs)
