from onnx_tf.common import exception
from onnx_tf.common import get_perm_from_formats
from onnx_tf.common import get_unique_suffix
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.frontend_handler import version


class SpaceToDepth(FrontendHandler):

  @classmethod
  def param_check(cls, node, **kwargs):
    if node.inputs[1] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[1], node.op)
    data_format = node.attr.get("data_format", "NHWC").decode()
    if data_format not in ["NHWC", "NCHW"]:
      exception.OP_UNSUPPORTED_EXCEPT("{} with data_format {}".format(
          node.op, data_format), "ONNX")

  @classmethod
  @version(1)
  def version_1(cls, node, **kwargs):
    blocksize = node.attr["block_size"]
    data_format = node.attr.get("data_format", "NHWC").decode()

    if data_format == "NHWC":
      transpose_unique_suffix = get_unique_suffix()
      space_to_depth_unique_suffix = get_unique_suffix()
      transpose_name = node.inputs[0] + "_T_" + transpose_unique_suffix
      space_to_depth_name = node.inputs[0] + "_T_STD_" + space_to_depth_unique_suffix
      before_transpose_node = cls.make_node(
          node, [node.inputs[0]], [transpose_name],
          perm=get_perm_from_formats(data_format, "NCHW"),
          name="Transpose")
      space_to_depth_node = cls.make_node(
          node, [transpose_name], [space_to_depth_name], blocksize=blocksize)
      after_transpose_node = cls.make_node(
          node, [space_to_depth_name],
          perm=get_perm_from_formats("NCHW", data_format),
          name="Transpose")
      return [before_transpose_node, space_to_depth_node, after_transpose_node]

    return cls.make_node(node, [node.inputs[0]], blocksize=blocksize)
