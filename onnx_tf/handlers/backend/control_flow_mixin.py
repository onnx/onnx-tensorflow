class LogicalMixin(object):

  @classmethod
  def process_attrs(cls, attrs):
    return cls._process_attrs(attrs, remove=["axis", "broadcast"])


class ComparisonMixin(object):

  @classmethod
  def process_attrs(cls, attrs):
    return cls._process_attrs(attrs, remove=["axis", "broadcast"])
