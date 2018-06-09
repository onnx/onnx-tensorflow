import copy

from .broadcast_mixin import BroadcastMixin


class BasicMathMixin(BroadcastMixin):
  pass


class ArithmeticMixin(BroadcastMixin):
  pass


class ReductionMixin(BroadcastMixin):

  @classmethod
  def _common(cls, node, **kwargs):
    attrs = copy.deepcopy(node.attrs)
    axis = attrs.pop("axes", None)
    if isinstance(axis, (list, tuple)) and len(axis) == 1:
      axis = axis[0]
    attrs["axis"] = axis
    # https://github.com/onnx/onnx/issues/585
    attrs["keepdims"] = attrs.pop("keepdims", 1) == 1
    return [cls.make_tensor_from_onnx_node(node, attrs=attrs, **kwargs)]
