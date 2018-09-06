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
  Finally, use ONNX operator name if above does not work.
  """

  @classmethod
  def check_cls(cls):
    super(FrontendHandler, cls).check_cls()
    if not cls.TF_OP:
      warnings.warn(
          "{} doesn't have TF_OP. "
          "Please use Handler.tf_op decorator to register TF_OP.".format(
              cls.__name__))

  @classmethod
  def make_node(cls,
                op_type,
                inputs,
                outputs,
                name=None,
                doc_string=None,
                version=0,
                should_check=True,
                **kwargs):
    """ Make a NodeProto from scratch.
    The main api is same to onnx.helper.make_node without any default value.

    :param op_type: The name of the operator to construct.
    :param inputs: Inputs names.
    :param outputs: Outputs names.
    :param name: optional unique identifier.
    :param doc_string: optional documentation string.
    :param version: Version used for check node. Default is cls.VERSION.
    :param should_check: Should check flag.
    Should set to False if is an unimplemented customized op.
    :param kwargs: Other args.
    :return: NodeProto.
    """
    node = helper.make_node(op_type, inputs, outputs, name, doc_string,
                            **kwargs)
    if should_check:
      cls.check_node(node, version)
    else:
      warnings.warn("Skipped check for {}.".format(node.op_type))
    return node

  @classmethod
  def make_node_from_tf_node(cls,
                             node,
                             inputs=None,
                             outputs=None,
                             op_type=None,
                             name=None,
                             doc_string=None,
                             version=0,
                             should_check=True,
                             **kwargs):
    """ Helper method to make node.
    The main api is almost same to onnx.helper.make_node with default value
    from TensorflowNode given.

    :param node: TensorflowNode object.
    :param inputs: Inputs names. Default is node.inputs.
    :param outputs: Outputs name. Default is cls.get_outputs_names(node).
    :param op_type: ONNX op name. Default is cls.ONNX_OP.
    :param name: Node name. Default is node.name.
    :param doc_string: optional documentation string.
    :param version: Version used for check node. Default is cls.VERSION.
    :param should_check: Should check flag.
    Should set to False if is an unimplemented customized op.
    :param kwargs: Other args.
    :return: NodeProto.
    """
    inputs = inputs if inputs is not None else node.inputs
    outputs = outputs if outputs is not None else cls.get_outputs_names(node)
    node = helper.make_node(
        op_type if op_type is not None else cls.ONNX_OP,
        inputs,
        outputs,
        name=name if name is not None else node.name,
        doc_string=doc_string,
        **kwargs)
    if should_check:
      cls.check_node(node, version)
    else:
      warnings.warn("Skipped check for {}.".format(node.op_type))
    return node

  @classmethod
  def check_node(cls, node, version=0):
    version = version or cls.VERSION
    if version == 0:
      raise ValueError("version can not be 0.")
    ctx = checker.C.CheckerContext()
    ctx.ir_version = onnx.IR_VERSION
    ctx.opset_imports = {cls.DOMAIN: version}
    checker.check_node(node, ctx=ctx)

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
