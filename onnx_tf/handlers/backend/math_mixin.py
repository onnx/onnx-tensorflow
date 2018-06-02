import copy

from .broadcast_mixin import BroadcastMixin


class BasicMathMixin(BroadcastMixin):

  @classmethod
  def process_attrs(cls, attrs):
    return cls._process_attrs(attrs, remove=["consumed_inputs"])


class ArithmeticMixin(BroadcastMixin):

  @classmethod
  def process_attrs(cls, attrs):
    return cls._process_attrs(
        attrs, remove=["consumed_inputs", "axis", "broadcast"])


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
    return [cls.make_tf_tensor(node, attrs=attrs, **kwargs)]
