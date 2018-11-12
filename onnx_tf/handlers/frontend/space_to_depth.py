from onnx_tf.common import exception
from onnx_tf.common import get_perm_from_formats
from onnx_tf.common import get_unique_suffix
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("SpaceToDepth")
@tf_op("SpaceToDepth")
class SpaceToDepth(FrontendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    data_format = node.attr.get("data_format", "NHWC").decode()
    if data_format not in ["NHWC", "NCHW"]:
      exception.OP_UNSUPPORTED_EXCEPT(
          "{} with data_format {}".format(node.op_type, data_format), "ONNX")

  @classmethod
  def version_1(cls, node, **kwargs):
    blocksize = node.attr["block_size"]
    data_format = node.attr.get("data_format", "NHWC").decode()

    if data_format == "NHWC":
      transpose_unique_suffix = get_unique_suffix()
      space_to_depth_unique_suffix = get_unique_suffix()
      transpose_name = node.inputs[0] + "_T_" + transpose_unique_suffix
      space_to_depth_name = node.inputs[0] + "_T_STD_" + space_to_depth_unique_suffix
      before_transpose_node = cls.make_node_from_tf_node(
          node, [node.inputs[0]], [transpose_name],
          perm=get_perm_from_formats(data_format, "NCHW"),
          op_type="Transpose",
          name=transpose_name)
      space_to_depth_node = cls.make_node_from_tf_node(
          node, [transpose_name], [space_to_depth_name],
          blocksize=blocksize,
          name=space_to_depth_name)
      after_transpose_node = cls.make_node_from_tf_node(
          node, [space_to_depth_name],
          perm=get_perm_from_formats("NCHW", data_format),
          op_type="Transpose")
      return [before_transpose_node, space_to_depth_node, after_transpose_node]

    return cls.make_node_from_tf_node(
        node, [node.inputs[0]], blocksize=blocksize)
