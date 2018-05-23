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
  """ This class is base frontend handler class.
  All frontend operator handler class MUST inherit this class.
  In frontend, operator handler class's name should be pascal case of file name
  which should be snake case.
  It is best to use tf functions' names. e.g. tf.nn.avg_pool
  If there is a multiple mapping, e.g. tf.nn.conv1d, tf.nn.conv2d, tf.nn.conv3d,
  try find common one first. In this case, tf.nn.convolution.
  Finally, use ONNX name if above does not work.
  """

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
    """ Helper method to make node. Each operator handler should call this
    instead of call helper.make_node directly.

    :param node: TensorflowNode.
    :param inputs: Inputs names. Default is node.inputs.
    :param outputs: Outputs name. Default is cls.get_outputs_names(node).
    :param onnx_op: ONNX op name. Default is cls.ONNX_OP.
    :param name: Node name. Default is node.name.
    :param version: Version used for check node. Default is cls.VERSION.
    :param should_check: Should check flag.
    Should set to False if is an unimplemented customized op.
    :param kwargs: Other args.
    :return: NodeProto.
    """
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

  @classmethod
  def get_outputs_names(cls, node, num=None):
    """ Helper method to get outputs names.
    e.g. tf.split: [Split, Split:1, Split:2]

    :param node: TensorflowNode.
    :param num: Force to get `num` outputs names.
    :return: List of outputs names.
    """
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
