import tensorflow as tf
from tensorflow_probability import distributions as tfd

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import partial_support
from onnx_tf.handlers.handler import ps_description


@onnx_op("Bernoulli")
@partial_support(True)
@ps_description(
    "Bernoulli with float type seed will be converted to int type seed")
class Bernoulli(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    dtype = node.attrs.get("dtype", x.dtype)
    dist = tfd.Bernoulli(probs=x, dtype=dtype)
    if 'seed' in node.attrs:
      ret = dist.sample(seed=int(node.attrs.get('seed')))
    else:
      ret = dist.sample()
    return [tf.cast(tf.reshape(ret, x.shape), dtype)]

  @classmethod
  def version_15(cls, node, **kwargs):
    return cls._common(node, **kwargs)
