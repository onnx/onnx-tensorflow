from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings

import onnx
from onnx import checker
from onnx import helper

from .handler import Handler
from onnx_tf.common import deprecated
from onnx_tf.common import get_perm_from_formats
from onnx_tf.common import get_unique_suffix
from onnx_tf.pb_wrapper import TensorflowNode


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
  def handle(cls, node, **kwargs):
    return super(FrontendHandler, cls).handle(node, **kwargs)

  @classmethod
  def handle_node_proto(cls, node, **kwargs):
    return super(FrontendHandler, cls).handle(TensorflowNode(node), **kwargs)

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
                             data_format_auto_convert=False,
                             **kwargs):
    """ Helper method to make node.
    The main api is almost same to onnx.helper.make_node with default value
    from TensorflowNode given.

    :param node: TensorflowNode object.
    :param inputs: Inputs names. Default is node.inputs.
    :param outputs: Outputs name. Default is node.outputs.
    :param op_type: ONNX op name. Default is cls.ONNX_OP.
    :param name: Node name. Default is node.name.
    :param doc_string: optional documentation string.
    :param version: Version used for check node. Default is cls.VERSION.
    :param should_check: Should check flag.
    Should set to False if is an unimplemented customized op.
    :param data_format_auto_convert: Pre and post transpose if data format is channel last.
    :param kwargs: Other args.
    :return: NodeProto.
    """
    from .frontend.transpose import Transpose

    inputs = inputs if inputs is not None else node.inputs
    outputs = outputs if outputs is not None else node.outputs
    data_format = node.attr.get("data_format", b"").decode("UTF-8")
    need_transpose = data_format_auto_convert and data_format.find("C") not in (
        -1, 1)

    nodes = []

    if need_transpose:
      # Add pre transpose
      c_first_data_format = data_format[0] + "C" + data_format[1:-1]
      pre_unique_suffix = get_unique_suffix()
      pre_transpose_node = Transpose.handle_node_proto(
          helper.make_node(
              "Transpose", [node.inputs[0], "perm"],
              [node.inputs[0] + "_T_" + pre_unique_suffix],
              name=node.inputs[0] + "_T_" + pre_unique_suffix),
          consts={
              "perm": get_perm_from_formats(data_format, c_first_data_format)
          })
      nodes.append(pre_transpose_node)
      inputs[0] = pre_transpose_node.output[0]

      # Process inputs, outputs name
      # Assume real input is always the first
      onnx_node_suffix = get_unique_suffix()
      onnx_node_output = node.outputs[0]
      inputs = [pre_transpose_node.output[0]] + inputs[1:]
      outputs = [onnx_node_output + "_" + onnx_node_suffix] + outputs[1:]

    onnx_node = helper.make_node(
        op_type if op_type is not None else cls.ONNX_OP,
        inputs,
        outputs,
        name=name if name is not None else node.name,
        doc_string=doc_string,
        **kwargs)

    if should_check:
      cls.check_node(onnx_node, version)
    else:
      warnings.warn("Skipped check for {}.".format(node.op_type))

    if need_transpose:
      nodes.append(onnx_node)
      # Add post transpose
      post_unique_suffix = get_unique_suffix()
      post_transpose_node = Transpose.handle_node_proto(
          helper.make_node(
              "Transpose", [onnx_node.output[0], "perm"], [onnx_node_output],
              name=onnx_node_output + "_" + onnx_node_suffix + "_T_" +
              post_unique_suffix),
          consts={
              "perm": get_perm_from_formats(c_first_data_format, data_format)
          })
      nodes.append(post_transpose_node)
      return nodes
    return onnx_node

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
  @deprecated("FrontendHandler.get_outputs_names is deprecated.{}. {}".format(
      deprecated.MSG_WILL_REMOVE, "Use node.outputs instead."))
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
