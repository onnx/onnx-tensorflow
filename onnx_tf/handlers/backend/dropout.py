import copy

import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Dropout")
@tf_func(tf.nn.dropout)
class Dropout(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]
    attrs = copy.deepcopy(node.attrs)

    if cls.SINCE_VERSION < 7 and attrs.pop("is_test", 0) == 0:
      attrs["keep_prob"] = 1 - attrs.pop("ratio", 0.5)
      return [cls.make_tensor_from_onnx_node(node, attrs=attrs, **kwargs)]
    elif cls.SINCE_VERSION < 12 : # for Opset 7, 10
      # at inference mode, is_test attribute is always set to 1
      # dropout at inference mode is a no-op
      return [x]
    else: # for Opset 12, 13
      # ratio and training_mode are optional and passed as inputs
      ratio = 0.5 # default ratio
      if len(node.inputs) > 1:
        ratio = tensor_dict[node.inputs[1]]
      training_mode = False # default is false
      if len(node.inputs) == 3:
        training_mode = tensor_dict[node.inputs[2]]

      return_mask = len(node.outputs) == 2 # if there are 2 outputs, mask is requested
      if ratio == 0 or training_mode is False: # Inferencing
        if return_mask is True:
          return x, tf.ones(x.shape, dtype=tf.bool)
        else:
          return [x]
      else: # Training
        # seed is passed in as an attribute
        seed = attrs.pop("seed", None)
        noise_shape = None # noise_shape is not passed in so default to None
        dropout_result = cls.make_tensor_from_onnx_node(node, inputs=[x, ratio, noise_shape, seed], attrs=attrs, **kwargs)
        if return_mask is True:
          # Create the mask based on the result of the Dropout
          mask = tf.dtypes.cast(dropout_result, tf.bool)
          return dropout_result, mask
        else:
          return [dropout_result]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_10(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_12(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
