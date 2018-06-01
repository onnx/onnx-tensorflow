import copy


class BasicMathMixin(object):

  @classmethod
  def process_attrs(cls, attrs):
    return cls._process_attrs(attrs, remove=["consumed_inputs"])


class ArithmeticMixin(object):

  @classmethod
  def process_attrs(cls, attrs):
    return cls._process_attrs(
        attrs, remove=["consumed_inputs", "axis", "broadcast"])


class ReductionMixin(object):

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
