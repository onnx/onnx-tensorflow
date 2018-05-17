from onnx_tf.handlers.frontend_handler import FrontendHandler


class BasicMathCommon(FrontendHandler):

  @classmethod
  def basic_math_op(cls, node, version, **kwargs):
    return cls.make_node(node, version=version)


class ArithmeticCommon(FrontendHandler):

  @classmethod
  def arithmetic_op(cls, node, version, **kwargs):
    axis = kwargs.get("axis", None)
    ex_kwargs = {}
    if axis is not None:
      ex_kwargs["axis"] = axis
    return cls.make_node(node, version=version, broadcast=1, **ex_kwargs)


class ReductionCommon(FrontendHandler):

  @classmethod
  def param_check(cls, node, version, **kwargs):
    if node.inputs[1] not in kwargs["consts"]:
      raise RuntimeError("axis of {} is not found in graph consts.".format(
          node.op))

  @classmethod
  def reduction_op(cls, node, version, **kwargs):
    consts = kwargs["consts"]
    axes = consts[node.inputs[1]]
    return cls.make_node(
        node, [node.inputs[0]],
        version=version,
        axes=axes,
        keepdims=node.attr.get("keep_dims", 1))
