from onnx_tf.handlers.frontend_handler import FrontendHandler


class Selu(FrontendHandler):
  _scale = 1.0507009873554804934193349852946
  _scale_alpha = 1.7580993408473768599402175208123
  _alpha = _scale_alpha / _scale  # 1.6732632423543774

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.make_node(node, version=1, gamma=cls._scale, alpha=cls._alpha)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.make_node(node, version=6, gamma=cls._scale, alpha=cls._alpha)
