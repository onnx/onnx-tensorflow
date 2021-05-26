import copy
import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func

@onnx_op("Multinomial")
@tf_func(tf.random.categorical)
class Multinomial(BackendHandler):

  @classmethod
  def version_7(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    sample_size = node.attrs.get("sample_size", 1)
    dtype = node.attrs.get("dtype", tf.int32)
    attrs = copy.deepcopy(node.attrs)
    attrs["num_samples"] = sample_size
    attrs["dtype"] = dtype
    return [cls.make_tensor_from_onnx_node(node, inputs=[x], attrs=attrs)]
