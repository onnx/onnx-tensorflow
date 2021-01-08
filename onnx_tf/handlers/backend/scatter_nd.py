import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from .gather_and_scatter_mixin import GatherAndScatterMixin


@onnx_op("ScatterND")
@tf_func(tf.tensor_scatter_nd_update)
class ScatterND(GatherAndScatterMixin, BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    data = kwargs["tensor_dict"][node.inputs[0]]
    indices = kwargs["tensor_dict"][node.inputs[1]]
    updates = kwargs["tensor_dict"][node.inputs[2]]

    result = cls.chk_idx_out_of_bounds(data, indices)
    msg = 'ScatterND indices are out of bounds, please double check the indices and retry.'
    with tf.control_dependencies(
        [tf.compat.v1.assert_equal(result, True, message=msg)]):
      indices = cls.process_neg_idx(data, indices)
      return [
          cls.make_tensor_from_onnx_node(node,
                                         inputs=[data, indices, updates],
                                         **kwargs)
      ]

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
