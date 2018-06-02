from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect

from onnx_tf.common import exception


class Handler(object):
  """ This class is base handler class.
  Base backend and frontend base handler class inherit this class.

  All operator handler MUST put decorator @onnx_op and @tf_op to register corresponding op.
  """

  ONNX_OP = None
  TF_OP = []

  DOMAIN = ""
  VERSION = 0
  SINCE_VERSION = 0

  @classmethod
  def check_cls(cls):
    if not cls.ONNX_OP:
      raise ValueError(
          "{} doesn't have ONNX_OP. "
          "Please use Handler.onnx_op decorator to register ONNX_OP.".format(
              cls.__name__))
    if not cls.TF_OP:
      raise ValueError(
          "{} doesn't have TF_OP. "
          "Please use Handler.tf_op decorator to register TF_OP.".format(
              cls.__name__))

  @classmethod
  def args_check(cls, node, **kwargs):
    """ Check args. e.g. if shape info is in graph.
    Raise exception if failed.

    :param node: NodeProto for backend or TensorflowNode for frontend.
    :param kwargs: Other args.
    """
    pass

  @classmethod
  def handle(cls, node, **kwargs):
    """ Main method in handler. It will find corresponding versioned handle method,
    whose name format is `version_%d`. So prefix `version_` is reserved in onnx-tensorflow.
    DON'T use it for other purpose.

    :param node: NodeProto for backend or TensorflowNode for frontend.
    :param kwargs: Other args.
    :return: NodeProto for frontend or TensorflowNode for backend.
    """
    ver_handle = getattr(cls, "version_{}".format(cls.SINCE_VERSION), None)
    if ver_handle:
      cls.args_check(node, **kwargs)
      return ver_handle(node, **kwargs)
    exception.OP_UNIMPLEMENTED_EXCEPT(node.op, cls.SINCE_VERSION)
    return None

  @classmethod
  def get_versions(cls):
    """ Get all support versions.

    :return: Version list.
    """
    versions = []
    for k, v in inspect.getmembers(cls, inspect.ismethod):
      if k.startswith("version_"):
        versions.append(int(k.replace("version_", "")))
    return versions

  @staticmethod
  def onnx_op(op):
    return Handler.property_register("ONNX_OP", op)

  @staticmethod
  def tf_op(op):
    ops = op
    if not isinstance(ops, list):
      ops = [ops]
    return Handler.property_register("TF_OP", ops)

  @staticmethod
  def domain(d):
    return Handler.property_register("DOMAIN", d)

  @staticmethod
  def property_register(name, value):

    def deco(cls):
      setattr(cls, name, value)
      return cls

    return deco


domain = Handler.domain
onnx_op = Handler.onnx_op
tf_op = Handler.tf_op
property_register = Handler.property_register
