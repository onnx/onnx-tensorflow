from .broadcast_mixin import BroadcastMixin

class BasicMathMixin(object):

  @classmethod
  def process_attrs(cls, attrs):
    return cls._process_attrs(attrs, remove=["consumed_inputs"])


class ArithmeticMixin(object):

  @classmethod
  def process_attrs(cls, attrs):
    return cls._process_attrs(
        attrs, remove=["consumed_inputs", "axis", "broadcast"])

  @classmethod
  def _limited_broadcast(cls, node, **kwargs):
    if node.attrs.get("broadcast") == 1:
      inputs = BroadcastMixin.explicit_broadcast(node.inputs,
                                                 node.attrs.get("axis", None),
                                                 kwargs["tensor_dict"])
      return [cls.make_tf_tensor(node, inputs=inputs, **kwargs)]
    return [cls.make_tf_tensor(node, **kwargs)]


class ReductionMixin(object):

  @classmethod
  def process_attrs(cls, attrs):
    return cls._process_attrs(
        attrs, remove=["consumed_inputs"], defalut={"axis": 0})

  @classmethod
  def _common(cls, node, **kwargs):
    values = [kwargs["tensor_dict"][inp] for inp in node.inputs]
    return [cls.make_tf_tensor(node, inputs=values, **kwargs)]
