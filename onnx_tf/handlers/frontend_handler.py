from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx
from onnx import defs
from onnx import helper
from onnx import checker

from onnx_tf.common import exception


class FrontendHandler(object):
  _TF_OP = []
  _ONNX_OP = ""

  @classmethod
  def handle(cls, node, version, **kwargs):
    since_version = defs.get_schema(
        cls.get_onnx_op(), domain="",
        max_inclusive_version=version).since_version
    ver_handle = getattr(cls, "version_{}".format(since_version), None)
    if ver_handle:
      cls.param_check(node, version, **kwargs)
      return ver_handle(node, **kwargs)
    exception.OP_NOT_IMPL_EXCEPT(node.op, since_version)

  @classmethod
  def param_check(cls, node, version, **kwargs):
    pass

  @classmethod
  def get_onnx_op(cls):
    return cls._ONNX_OP or cls.__name__

  @classmethod
  def get_tf_op(cls):
    return cls._TF_OP or [cls.__name__]

  @classmethod
  def make_node(cls,
                node,
                inputs=None,
                outputs=None,
                version=None,
                should_check=True,
                **kwargs):
    inputs = inputs if inputs is not None else node.inputs
    outputs = outputs if outputs is not None else [
        node.name + ":{}".format(i) if i > 0 else node.name
        for i in range(len(node.attr["_output_shapes"]))
    ]
    node = helper.make_node(
        cls.get_onnx_op(), inputs, outputs, name=node.name, **kwargs)
    if should_check:
      if version is None:
        raise RuntimeError("version can not be None.")
      ctx = checker.C.CheckerContext()
      ctx.ir_version = onnx.IR_VERSION
      ctx.opset_imports = {"": version}
      checker.check_node(node, ctx=ctx)
    return node

  @classmethod
  def get_versions(cls):
    versions = []
    for name in dir(cls):
      if name.startswith("version_"):
        versions.append(int(name.replace("version_", "")))
    return versions
