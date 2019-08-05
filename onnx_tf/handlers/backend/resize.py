import tensorflow as tf

from onnx_tf.common import exception
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("Resize")
class Resize(BackendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    x_shape = x.get_shape().as_list()
    if len(x_shape) != 4:
      exception.OP_UNSUPPORTED_EXCEPT("Resize required 4D input", "Tensorflow")

  @classmethod
  def version_10(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    x_shape = x.get_shape().as_list()
    scales = kwargs["tensor_dict"][node.inputs[1]]

    n_in_scales_is_one = tf.equal(scales[0], 1)
    c_in_scales_is_one = tf.logical_or(
        tf.equal(scales[1], 1), tf.equal(scales[3], 1))
    assert_n_c_in_scales_are_ones = tf.Assert(
        tf.logical_and(n_in_scales_is_one, c_in_scales_is_one), [scales])

    with tf.control_dependencies([assert_n_c_in_scales_are_ones]):
      x_in_NCHW_format = tf.equal(scales[1], 1)
      h_w_scale = tf.where(x_in_NCHW_format, scales[2:], scales[1:3])
      h_w_shape = tf.where(x_in_NCHW_format, x_shape[2:], x_shape[1:3])
      new_h_w_shape = tf.cast(h_w_scale * tf.cast(h_w_shape, scales.dtype),
                              tf.int32)

      mode = node.attrs.get("mode", "nearest")
      if mode.lower() == "linear":
        mode = tf.image.ResizeMethod.BILINEAR
      else:
        mode = tf.image.ResizeMethod.NEAREST_NEIGHBOR

      def process_NCHW_format(x):
        x_t = tf.transpose(x, perm=[0, 2, 3, 1])
        y = tf.image.resize_images(x_t, size=new_h_w_shape, method=mode)
        y_t = tf.transpose(y, perm=[0, 3, 1, 2])
        return y_t

      def process_NHWC_format(x):
        y = tf.image.resize_images(x, size=new_h_w_shape, method=mode)
        return y

      output = tf.cond(x_in_NCHW_format, lambda: process_NCHW_format(x),
                       lambda: process_NHWC_format(x))

      return [output]
