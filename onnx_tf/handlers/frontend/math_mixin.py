import numpy as np

from onnx_tf.common import exception
from .broadcast_mixin import BroadcastMixin


class BasicMathMixin(object):

  @classmethod
  def basic_math_op(cls, node, **kwargs):
    return cls.make_node_from_tf_node(node)


class ArithmeticMixin(BroadcastMixin):

  @classmethod
  def arithmetic_op(cls, node, **kwargs):
    if cls.SINCE_VERSION <= 6:
      return cls.limited_broadcast(node, **kwargs)
    else:  # since_version >= 7
      return cls.np_broadcast(node, **kwargs)


class ReductionMixin(object):

  @classmethod
  def args_check(cls, node, **kwargs):
    if node.inputs[1] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[1], node.op_type)

  @classmethod
  def reduction_op(cls, node, **kwargs):
    consts = kwargs["consts"]
    axes = consts[node.inputs[1]]
    # Expand dim if axes is a 0-d array
    if len(np.shape(axes)) == 0:
      axes = np.expand_dims(axes, 0)
    return cls.make_node_from_tf_node(
        node, [node.inputs[0]],
        axes=axes,
        keepdims=node.attr.get("keep_dims", 1))
