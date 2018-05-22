from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx
from onnx import helper
from onnx import checker

from onnx_tf.common import exception
from .handler import Handler


class FrontendHandler(Handler):

  _cls_ver_handle = {}
  _cls_versions = {}

  @classmethod
  def param_check(cls, node, **kwargs):
    pass

  @classmethod
  def handle(cls, node, **kwargs):
    ver_handle = cls.get_ver_handle(cls.SINCE_VERSION)
    if ver_handle:
      cls.param_check(node, **kwargs)
      return ver_handle(cls, node, **kwargs)
    exception.OP_UNIMPLEMENTED_EXCEPT(node.op)
    return None

  @classmethod
  def make_node(cls,
                node,
                inputs=None,
                outputs=None,
                onnx_op=None,
                name=None,
                version=0,
                should_check=True,
                **kwargs):
    inputs = inputs if inputs is not None else node.inputs
    outputs = outputs if outputs is not None else cls.get_outputs_names(node)
    node = helper.make_node(
        onnx_op if onnx_op is not None else cls.get_onnx_op(),
        inputs,
        outputs,
        name=name if name is not None else node.name,
        **kwargs)
    if should_check:
      version = version or cls.VERSION
      if version == 0:
        raise RuntimeError("version can not be 0.")
      ctx = checker.C.CheckerContext()
      ctx.ir_version = onnx.IR_VERSION
      ctx.opset_imports = {cls.DOMAIN: version}
      checker.check_node(node, ctx=ctx)
    return node

  @classmethod
  def get_onnx_op(cls):
    return cls.ONNX_OP or cls.__name__

  @classmethod
  def get_tf_op(cls):
    return cls.TF_OP or [cls.__name__]

  @classmethod
  def get_outputs_names(cls, node, num=None):
    num = num or len(node.attr["_output_shapes"])
    return [
        node.name + ":{}".format(i) if i > 0 else node.name for i in range(num)
    ]


version = FrontendHandler.version
