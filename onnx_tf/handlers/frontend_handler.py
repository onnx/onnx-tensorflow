from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from functools import partial
import warnings

import onnx
from onnx import defs
from onnx import helper
from onnx import checker

from onnx_tf.common import exception


class FrontendHandler(object):
  _TF_OP = []
  _ONNX_OP = None
  DOMAIN = ""

  @classmethod
  def param_check(cls, node, version, **kwargs):
    pass

  @classmethod
  def handle(cls, node, version, **kwargs):
    since_version = 1
    # TODO(fumihwh): Use defs.has(cls.get_onnx_op(), domain=cls.DOMAIN)
    schema_version_map = defs.C.schema_version_map()
    if cls.DOMAIN in schema_version_map and cls.get_onnx_op(
    ) in schema_version_map[cls.DOMAIN]:
      since_version = defs.get_schema(
          cls.get_onnx_op(), domain=cls.DOMAIN,
          max_inclusive_version=version).since_version
    else:
      warnings.warn("Unknown op {} in domain `{}`.".format(
          cls.get_onnx_op(), cls.DOMAIN or "ai.onnx"))
    ver_handle = getattr(cls, "version_{}".format(since_version), None)
    if ver_handle:
      cls.param_check(node, version, **kwargs)
      cls.make_node = partial(cls.make_node, version=version)
      return ver_handle(node, **kwargs)
    exception.OP_UNIMPLEMENTED_EXCEPT(node.op)
    return None

  @classmethod
  def make_node(cls,
                node,
                inputs=None,
                outputs=None,
                onnx_op=None,
                name=None,
                version=None,  # preset by `handle`
                should_check=True,
                **kwargs):
    inputs = inputs if inputs is not None else node.inputs
    outputs = outputs if outputs is not None else cls.get_outputs_names(node)
    node = helper.make_node(
        onnx_op or cls.get_onnx_op(),
        inputs,
        outputs,
        name=name or node.name,
        **kwargs)
    if should_check:
      if version is None:
        raise RuntimeError("version can not be None.")
      ctx = checker.C.CheckerContext()
      ctx.ir_version = onnx.IR_VERSION
      ctx.opset_imports = {cls.DOMAIN: version}
      checker.check_node(node, ctx=ctx)
    return node

  @classmethod
  def get_onnx_op(cls):
    return cls._ONNX_OP or cls.__name__

  @classmethod
  def get_tf_op(cls):
    return cls._TF_OP or [cls.__name__]

  @classmethod
  def get_versions(cls):
    versions = []
    for method_name in dir(cls):
      if method_name.startswith("version_"):
        versions.append(int(method_name.replace("version_", "")))
    return versions

  @classmethod
  def get_outputs_names(cls, node, num=None):
    num = num or len(node.attr["_output_shapes"])
    return [
        node.name + ":{}".format(i) if i > 0 else node.name for i in range(num)
    ]
