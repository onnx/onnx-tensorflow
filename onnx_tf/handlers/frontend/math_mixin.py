from onnx_tf.common import exception
from onnx_tf.common.broadcast import get_broadcast_axis


class BasicMathMixin(object):

  @classmethod
  def basic_math_op(cls, node, **kwargs):
    return cls.make_node_from_tf_node(node)


class ArithmeticMixin(object):

  @classmethod
  def arithmetic_op(cls, node, **kwargs):
    if cls.SINCE_VERSION <= 6:
      return cls._limited_broadcast(node, **kwargs)
    else:  # since_version >= 7
      return cls._np_broadcast(node, **kwargs)

  @classmethod
  def _limited_broadcast(cls, node, **kwargs):
    node_dict = kwargs["node_dict"]
    axis = kwargs.get(
        "axis", get_broadcast_axis(*[node_dict[x] for x in node.inputs[0:2]]))
    ex_kwargs = {}
    if axis is not None:
      ex_kwargs["axis"] = axis
    return cls.make_node_from_tf_node(node, broadcast=1, **ex_kwargs)

  @classmethod
  def _np_broadcast(cls, node, **kwargs):
    return cls.make_node(node)


class ReductionMixin(object):

  @classmethod
  def args_check(cls, node, **kwargs):
    if node.inputs[1] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[1], node.op)

  @classmethod
  def reduction_op(cls, node, **kwargs):
    consts = kwargs["consts"]
    axes = consts[node.inputs[1]]
    return cls.make_node_from_tf_node(
        node, [node.inputs[0]],
        axes=axes,
        keepdims=node.attr.get("keep_dims", 1))
