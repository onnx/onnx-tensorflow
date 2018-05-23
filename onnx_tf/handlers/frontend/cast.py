import numpy as np
from onnx import mapping
import tensorflow as tf

from onnx_tf.handlers.frontend_handler import FrontendHandler


class Cast(FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    dst_t = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(
        tf.as_dtype(node.attr["DstT"]).as_numpy_dtype)]
    return cls.make_node(node, [node.inputs[0]], to=dst_t)
