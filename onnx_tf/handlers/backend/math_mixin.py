from onnx_tf.common import exception


class BasicMathMixin(object):

  @classmethod
  def process_attrs(cls, attrs):
    attrs.pop("consumed_inputs", None)
    return attrs


class ArithmeticMixin(object):

  @classmethod
  def process_attrs(cls, attrs):
    attrs.pop("axis", None)
    attrs.pop("broadcast", None)
    attrs.pop("consumed_inputs", None)
    return attrs


class ReductionMixin(object):

  @classmethod
  def args_check(cls, node, **kwargs):
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
