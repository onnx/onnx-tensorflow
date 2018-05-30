from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("Transpose")
@tf_op("Transpose")
class Transpose(FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    consts = kwargs["consts"]
    if node.inputs[1] in consts:
      perm = consts[node.inputs[1]]
    else:
      input_rank = len(
          kwargs['node_dict'][node.inputs[0]].attr['_output_shapes'][0])
      perm = list(reversed(range(input_rank)))

    return cls.make_node_from_tf_node(node, [node.inputs[0]], perm=perm)
