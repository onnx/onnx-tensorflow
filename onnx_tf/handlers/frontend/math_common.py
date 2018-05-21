from onnx_tf.common import exception
from onnx_tf.common.broadcast import get_broadcast_axis
from onnx_tf.handlers.frontend_handler import FrontendHandler


class BasicMathCommon(FrontendHandler):

  @classmethod
  def basic_math_op(cls, node, **kwargs):
    return cls.make_node(node)


class ArithmeticCommon(FrontendHandler):

  @classmethod
  def arithmetic_op(cls, node, **kwargs):
    node_dict = kwargs["node_dict"]
    axis = kwargs.get(
        "axis", get_broadcast_axis(*[node_dict[x] for x in node.inputs[0:2]]))
    ex_kwargs = {}
    if axis is not None:
      ex_kwargs["axis"] = axis
    return cls.make_node(node, broadcast=1, **ex_kwargs)


class ReductionCommon(FrontendHandler):

  @classmethod
  def param_check(cls, node, **kwargs):
    if node.inputs[1] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[1], node.op)

  @classmethod
  def reduction_op(cls, node, **kwargs):
    consts = kwargs["consts"]
    axes = consts[node.inputs[1]]
    return cls.make_node(
        node, [node.inputs[0]],
        axes=axes,
        keepdims=node.attr.get("keep_dims", 1))
