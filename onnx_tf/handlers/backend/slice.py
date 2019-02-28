import numpy as np
import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Slice")
@tf_func(tf.slice)
class Slice(BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]

    full_sizes = x.get_shape().as_list()
    full_begin = [0] * len(full_sizes)

    starts = node.attrs.get("starts")
    ends = node.attrs.get("ends")
    slice_len = len(starts)
    axes = node.attrs.get("axes", list(range(slice_len)))

    for i in range(slice_len):
      starts[i] = full_sizes[axes[i]] + starts[i] if starts[i] < 0 else starts[
          i]
      ends[i] = full_sizes[axes[i]] + ends[i] if ends[i] < 0 else ends[i]
      if full_sizes[axes[i]] is not None:
        ends[i] = np.min([full_sizes[axes[i]], ends[i]])
        starts[i] = np.min([full_sizes[axes[i]], starts[i]])
      full_begin[axes[i]] = starts[i]
      full_sizes[axes[i]] = ends[i] - starts[i]

    return [
        cls.make_tensor_from_onnx_node(
            node,
            inputs=[
                tensor_dict[node.inputs[0]],
                tf.constant(full_begin),
                tf.constant(full_sizes)
            ],
            **kwargs)
    ]
