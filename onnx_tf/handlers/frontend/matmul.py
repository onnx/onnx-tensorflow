from onnx_tf.common import get_unique_suffix
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.frontend_handler import version


class Matmul(FrontendHandler):
  TF_OP = ["MatMul"]
  ONNX_OP = "MatMul"

  @classmethod
  @version(1)
  def version_1(cls, node, **kwargs):
    transpose_a = node.attr.get("transpose_a", False)
    transpose_b = node.attr.get("transpose_b", False)
    input_a = node.inputs[0]
    input_b = node.inputs[1]
    nodes = []
    if transpose_a:
      unique_suffix_a = get_unique_suffix()
      transposed_a = cls.make_node(
          node, [node.inputs[0]], [node.inputs[0] + "_T_" + unique_suffix_a],
          onnx_op="Transpose",
          name=node.inputs[0] + "_T_" + unique_suffix_a)
      input_a = node.inputs[0] + "_T_" + unique_suffix_a
      nodes.append(transposed_a)
    if transpose_b:
      unique_suffix_b = get_unique_suffix()
      transposed_b = cls.make_node(
          node, [node.inputs[1]], [node.inputs[1] + "_T_" + unique_suffix_b],
          onnx_op="Transpose",
          name=node.inputs[1] + "_T_" + unique_suffix_b)
      input_b = node.inputs[1] + "_T_" + unique_suffix_b
      nodes.append(transposed_b)
    nodes.append(cls.make_node(node, [input_a, input_b]))
    return nodes
