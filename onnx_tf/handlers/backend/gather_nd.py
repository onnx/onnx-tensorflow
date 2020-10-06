import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from .gather_and_scatter_mixin import GatherAndScatterMixin


@onnx_op("GatherND")
@tf_func(tf.gather_nd)
class GatherND(GatherAndScatterMixin, BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    data = kwargs["tensor_dict"][node.inputs[0]]
    indices = kwargs["tensor_dict"][node.inputs[1]]
    batch_dims = node.attrs.get('batch_dims', 0)

    result = cls.chk_idx_out_of_bounds(data, indices, batch_dims)
    msg = 'GatherND indices are out of bounds, please double check the indices and retry.'
    with tf.control_dependencies(
        [tf.compat.v1.assert_equal(result, True, message=msg)]):
      indices = cls.process_neg_idx(data, indices, batch_dims)
      return [
          cls.make_tensor_from_onnx_node(node, inputs=[data, indices], **kwargs)
      ]

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_12(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
