import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from .gather_and_scatter_mixin import GatherAndScatterMixin


@onnx_op("GatherND")
class GatherND(GatherAndScatterMixin, BackendHandler):

  @classmethod
  def version_11(cls, node, **kwargs):
    data = kwargs["tensor_dict"][node.inputs[0]]
    indices = kwargs["tensor_dict"][node.inputs[1]]

    result = cls.chk_idx_out_of_bounds(data, indices)
    msg = 'GatherND indices are out of bounds, please double check the indices and retry.'
    with tf.control_dependencies([tf.compat.v1.assert_equal(result, True, message=msg)]):
      indices = cls.process_neg_idx(data, indices)
      return [tf.gather_nd(data, indices)]
