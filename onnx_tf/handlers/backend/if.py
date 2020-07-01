import tensorflow as tf

import onnx_tf
from onnx.helper import make_opsetid
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("If")
@tf_func(tf.cond)
class If(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    cond = kwargs["tensor_dict"][node.inputs[0]]
    then_branch = node.attrs["then_branch"]
    else_branch = node.attrs["else_branch"]
    current_opset = [make_opsetid(cls.DOMAIN, cls.VERSION)]

    def true_fn():
      subgraph_tensor_dict = onnx_tf.backend.onnx_graph_to_tensorflow_ops(
          subgraph=then_branch,
          input_values={}, # all inputs of then_branch are in tensor_dict
          tensor_dict=kwargs["tensor_dict"],
          opset=current_opset)
      return [subgraph_tensor_dict[o.name] for o in then_branch.output]

    def false_fn():
      subgraph_tensor_dict = onnx_tf.backend.onnx_graph_to_tensorflow_ops(
          subgraph=else_branch,
          input_values={}, # all inputs of else_branch are in tensor_dict
          tensor_dict=kwargs["tensor_dict"],
          opset=current_opset)
      return [subgraph_tensor_dict[o.name] for o in else_branch.output]

    return [
        cls.make_tensor_from_onnx_node(node, inputs=[cond, true_fn, false_fn])
    ]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)
