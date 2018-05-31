from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .handler import Handler


# TODO(fumihwh): fix all doc
class BackendHandler(Handler):
  """ This class is base frontend handler class.
  All frontend operator handler class MUST inherit this class.
  In frontend, operator handler class's name should be pascal case of file name
  which should be snake case.
  It is best to use tf functions' names. e.g. tf.nn.avg_pool
  If there is a multiple mapping, e.g. tf.nn.conv1d, tf.nn.conv2d, tf.nn.conv3d,
  try find common one first. In this case, tf.nn.convolution.
  Finally, use ONNX name if above does not work.
  """

  TF_FUNC = None

  @classmethod
  def process_attrs(cls, attrs):
    return attrs

  # @classmethod
  # def check_cls(cls):
  #   super(BackendHandler, cls).check_cls()
  #   if not cls.TF_FUNC:
  #     raise ValueError(
  #         "{} doesn't have TF_FUNC. "
  #         "Please use Handler.tf_func decorator to register TF_FUNC.".format(
  #             cls.__name__))

  @classmethod
  def make_tf_tensor(cls, node, tf_func=None, inputs=None, attrs=None, name=None, **kwargs):
    tensor_dict = kwargs.pop("tensor_dict", {})
    tf_func = tf_func or cls.TF_FUNC
    inputs = inputs or [tensor_dict.get(inp, None) for inp in node.inputs]
    attrs = attrs or cls.process_attrs(node.attrs)
    name = name or node.name
    if name != "":
      attrs["name"] = name
    return tf_func(*inputs, **attrs)
