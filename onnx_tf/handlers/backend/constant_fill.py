import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("ConstantFill")
@tf_func(tf.fill)
class ConstantFill(BackendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    if node.inputs and "shape" in node.attrs:
      raise ValueError(
          "Cannot set the shape argument and pass in an input at the same time."
      )
    if not node.inputs and "extra_shape" in node.attrs:
      raise ValueError("Cannot set extra_shape when there is no input.")

  @classmethod
  def process_attrs(cls, attrs):
    return cls._process_attrs(
        attrs,
        remove=["shape", "input_as_shape", "extra_shape", "dtype"],
        default={"value": 0.})

  @classmethod
  def version_1(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]

    if "shape" in node.attrs:
      shape = node.attrs["shape"]
    else:
      shape = tensor_dict[
          node.inputs[0]].get_shape().as_list() if node.attrs.get(
              "input_as_shape", 0) == 0 else tensor_dict[node.inputs[0]]

    if "extra_shape" in node.attrs:
      shape = tf.concat([shape, node.attrs["extra_shape"]], 0)

    value = node.attrs.get("value", 0.)

    if "dtype" in node.attrs:
      return [tf.cast(tf.fill(shape, value), dtype=node.attrs["dtype"])]

    return [cls.make_tf_tensor(node, inputs=[shape], **kwargs)]
