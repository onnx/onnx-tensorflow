from .control_flow_common import ComparisonCommon


class Equal(ComparisonCommon):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.comparison_op(node, **kwargs)
