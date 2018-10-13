from onnx_tf.common import get_unique_suffix
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("MatMul")
@tf_op("MatMul")
class Matmul(FrontendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    transpose_a = node.attr.get("transpose_a", False)
    transpose_b = node.attr.get("transpose_b", False)
    input_a = node.inputs[0]
    input_b = node.inputs[1]
    nodes = []
    if transpose_a:
      unique_suffix_a = get_unique_suffix()
      transposed_a = cls.make_node_from_tf_node(
          node, [node.inputs[0]], [node.inputs[0] + "_T_" + unique_suffix_a],
          op_type="Transpose",
          name=node.inputs[0] + "_T_" + unique_suffix_a)
      input_a = node.inputs[0] + "_T_" + unique_suffix_a
      nodes.append(transposed_a)
    if transpose_b:
      unique_suffix_b = get_unique_suffix()
      transposed_b = cls.make_node_from_tf_node(
          node, [node.inputs[1]], [node.inputs[1] + "_T_" + unique_suffix_b],
          op_type="Transpose",
          name=node.inputs[1] + "_T_" + unique_suffix_b)
      input_b = node.inputs[1] + "_T_" + unique_suffix_b
      nodes.append(transposed_b)
    nodes.append(cls.make_node_from_tf_node(node, [input_a, input_b]))
    return nodes

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_9(cls, node, **kwargs):
    return cls._common(node, **kwargs)
