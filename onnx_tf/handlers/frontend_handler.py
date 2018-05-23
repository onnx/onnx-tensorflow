from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings

import onnx
from onnx import helper
from onnx import checker

from .handler import Handler


class FrontendHandler(Handler):

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
        onnx_op if onnx_op is not None else cls.ONNX_OP,
        inputs,
        outputs,
        name=name if name is not None else node.name,
        **kwargs)
    if should_check:
      version = version or cls.VERSION
      if version == 0:
        raise ValueError("version can not be 0.")
      ctx = checker.C.CheckerContext()
      ctx.ir_version = onnx.IR_VERSION
      ctx.opset_imports = {cls.DOMAIN: version}
      checker.check_node(node, ctx=ctx)
    else:
      warnings.warn("Skipped check for {}.".format(node.op_type))
    return node
