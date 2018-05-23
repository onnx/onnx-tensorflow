from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import warnings

from onnx_tf.common import exception


class Handler(object):

  ONNX_OP = None
  TF_OP = []

  DOMAIN = ""
  VERSION = 0
  SINCE_VERSION = 0

  @classmethod
  def check(cls):
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
    pass

  @classmethod
  def handle(cls, node, **kwargs):
    ver_handle = getattr(cls, "version_{}".format(cls.SINCE_VERSION), None)
    if ver_handle:
      cls.args_check(node, **kwargs)
      return ver_handle(node, **kwargs)
    exception.OP_UNIMPLEMENTED_EXCEPT(node.op)
    return None

  @classmethod
  def get_versions(cls):
    versions = []
    for k, v in inspect.getmembers(cls, inspect.ismethod):
      if k.startswith("version_"):
        versions.append(int(k.replace("version_", "")))
    return versions

  @classmethod
  def get_outputs_names(cls, node, num=None):
    if num is None:
      if "_output_shapes" in node.attr:
        num = len(node.attr["_output_shapes"])
      else:
        num = 1
        warnings.warn("_output_shapes is not in node.attr. "
                      "The num of output is set to 1 for commonly. "
                      "It will cause problem with case of multiple outputs.")
    return [
        node.name + ":{}".format(i) if i > 0 else node.name for i in range(num)
    ]

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
