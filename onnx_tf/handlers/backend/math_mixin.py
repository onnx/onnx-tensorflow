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
    x = kwargs["tensor_dict"][node.inputs[0]]
    attrs = copy.deepcopy(node.attrs)
    attrs["axis"] = attrs.pop("axes", list(range(len(x.get_shape().as_list()))))
    # https://github.com/onnx/onnx/issues/585
    attrs["keepdims"] = attrs.pop("keepdims", 1) == 1
    return [cls.make_tf_tensor(node, attrs=attrs, **kwargs)]
