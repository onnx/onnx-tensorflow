import tensorflow as tf

from onnx_tf.common import exception


class MathMixin(object):

  @classmethod
  def explicit_broadcast(cls, inputs, axis, tensor_dict):
    x = tensor_dict[inputs[0]]
    y = tensor_dict[inputs[1]]

    if axis is None:
      return [x, y]

    total_num_dim = len(x.get_shape())
    if axis < 0:
      axis += total_num_dim

    if axis + len(y.get_shape()) == total_num_dim:
      return [x, y]

    dims = [axis + i for i in range(len(y.get_shape()))]
    for i in range(total_num_dim):
      if i not in dims:
        new_y = tf.expand_dims(y, i)
    return [x, new_y]


class BasicMathMixin(MathMixin):

  @classmethod
  def process_attrs(cls, attrs):
    attrs.pop("consumed_inputs", None)
    return attrs


class ArithmeticMixin(MathMixin):

  @classmethod
  def process_attrs(cls, attrs):
    return attrs


class ReductionMixin(MathMixin):

  @classmethod
  def args_check(cls, node, **kwargs):
    if node.inputs[1] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[1], node.op_type)

  @classmethod
  def reduction_op(cls, node, **kwargs):
    consts = kwargs["consts"]
    axes = consts[node.inputs[1]]
    return cls.make_node(
        node, [node.inputs[0]],
        axes=axes,
        keepdims=node.attr.get("keep_dims", 1))
