import tensorflow as tf

from onnx_tf.common import exception
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import partial_support
from onnx_tf.handlers.handler import ps_description
from onnx_tf.common.tf_helper import tf_shape


@onnx_op("Resize")
@partial_support(True)
@ps_description(
    "Resize required 4D input in Tensorflow. " +
    "For opset 11, only the following attributes and inputs " +
    "conbination are supported in Tensorflow:\n\t1. mode=nearest, " +
    "coordinate_transformation_mode=align_corners, " +
    "nearest_mode=round_prefer_ceil, can use scales(*) or sizes.\n" +
    "\t2. mode=nearest, coordinate_transformation_mode=asymmetric, " +
    "nearest_mode=floor, can use scales(*) or sizes.\n\t3. mode=" +
    "nearest, coordinate_transformation_mode=tf_half_pixel_for_nn, " +
    "nearest_mode=floor, can use scales(*) or sizes.\n\t4. mode=" +
    "linear, coordinate_transformation_mode=align_corners, " +
    "can use scales(*) or sizes.\n\t5. mode=linear, coordinate_" +
    "transformation_mode=asymmetric, can use scales(*) or sizes.\n" +
    "\t6. mode=linear, coordinate_transformation_mode=half_pixel, " +
    "can use scales(*) or sizes.\n\t7. mode=cubic, coordinate_" +
    "transformation_mode=align_corners, cubic_coeff_a=-0.5, " +
    "exclude_outside=1, can use scales(*) or sizes.\n\t8. mode=" +
    "cubic, coordinate_transformation_mode=asymmetric, " +
    "cubic_coeff_a=-0.5, exclude_outside=1, can use scales(*) " +
    "or sizes.\n\t9. mode=cubic, coordinate_transformation_mode=" +
    "half_pixel, cubic_coeff_a=-0.5, exclude_outside=1, " +
    "can use scales(*) or sizes.\n\t10. mode=nearest, " +
    "coordinate_transformation_mode=tf_crop_and_resize, " +
    "extrapolation_value=any_float_value, nearest_mode=" +
    "round_prefer_ceil, can use scales or sizes.\n\t11. mode=linear, " +
    "coordinate_transformation_mode=tf_crop_and_resize, " +
    "extrapolation_value=any_float_value, can use scales or sizes." +
    "\n\t- Note (*): The accuracy of your model will go down, if the height and " +
    "the width of the new sizes(scales * origial sizes) are not in whole numbers."
)
class Resize(BackendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    x_shape = x.get_shape().as_list()
    if len(x_shape) != 4:
      exception.OP_UNSUPPORTED_EXCEPT("Resize required 4D input", "Tensorflow")
    if cls.SINCE_VERSION >= 11:
      # supported attributes combination
      # ____________________________________________________________________________________________________________________________________________________
      # | mode    | coordinate_transformation_mode | cubic_coeff_a | exclude_outside | extrapolation_value | nearest_mode      | scales        | sizes     |
      # |_________|________________________________|_______________|_________________|_____________________|___________________|_______________|___________|
      # | nearest | align_corners                  | not apply     | 0               | not apply           | round_prefer_ceil | supported (1) | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | nearest | asymmetric                     | not apply     | 0               | not apply           | floor             | supported (1) | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | nearest | tf_half_pixel_for_nn           | not apply     | 0               | not apply           | floor             | supported (1) | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | linear  | align_corners                  | not apply     | 0               | not apply           | not apply         | supported (1) | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | linear  | asymmetric                     | not apply     | 0               | not apply           | not apply         | supported (1) | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | linear  | half_pixel                     | not apply     | 0               | not apply           | not apply         | supported (1) | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | cubic   | align_corners                  | -0.5          | 1               | not apply           | not apply         | supported (1) | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | cubic   | asymmetric                     | -0.5          | 1               | not apply           | not apply         | supported (1) | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | cubic   | half_pixel                     | -0.5          | 1               | not apply           | not apply         | supported (1) | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | nearest | tf_crop_and_resize             | not apply     | 0               | any float value     | round_prefer_ceil | supported     | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | linear  | tf_crop_and_resize             | not apply     | 0               | any float value     | not apply         | supported     | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # Note:
      # 1. The accuracy of your model will go down, if the height and the width of the new sizes(scales * origial sizes) are not in whole numbers.
      coordinate_transformation_mode = node.attrs.get(
          "coordinate_transformation_mode", "half_pixel")
      cubic_coeff_a = node.attrs.get("cubic_coeff_a", -0.75)
      exclude_outside = node.attrs.get("exclude_outside", 0)
      mode = node.attrs.get("mode", "nearest")
      nearest_mode = node.attrs.get("nearest_mode", "round_prefer_floor")
      if coordinate_transformation_mode == "pytorch_half_pixel":
        exception.OP_UNSUPPORTED_EXCEPT(
            "Resize coordinate_transformation_mode=pytorch_half_pixel",
            "Tensorflow")
      if (coordinate_transformation_mode == "half_pixel" and mode == "nearest"
         ) or (coordinate_transformation_mode == "tf_half_pixel_for_nn" and
               mode in ["linear", "cubic"]) or (
                   coordinate_transformation_mode == "tf_crop_and_resize" and
                   mode == "cubic"):
        exception.OP_UNSUPPORTED_EXCEPT(
            "Resize coordinate_transformation_mode=" +
            coordinate_transformation_mode + " and  mode=" + mode, "Tensorflow")
      if (exclude_outside == 1 and
          mode in ["nearest", "linear"]) or (exclude_outside == 0 and
                                             mode == "cubic"):
        exception.OP_UNSUPPORTED_EXCEPT(
            "Resize mode=" + mode + " and exclude_outside=" +
            str(exclude_outside), "Tensorflow")
      if cubic_coeff_a != -0.5 and mode == "cubic":
        exception.OP_UNSUPPORTED_EXCEPT(
            "Resize mode=cubic and cubic_coeff_a=" + cubic_coeff_a,
            "Tensorflow")
      if mode == "nearest":
        if (nearest_mode in [
            "round_prefer_floor", "ceil"
        ]) or (coordinate_transformation_mode in [
            "align_corners", "tf_crop_and_resize"
        ] and nearest_mode == "floor") or (coordinate_transformation_mode in [
            "asymmetric", "tf_half_pixel_for_nn"
        ] and nearest_mode == "round_prefer_ceil"):
          exception.OP_UNSUPPORTED_EXCEPT(
              "Resize coordinate_transformation_mode=" +
              coordinate_transformation_mode +
              ", mode=nearest and nearest_mode=" + nearest_mode, "Tensorflow")

  @classmethod
  def version_10(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    x_shape = tf_shape(x)
    scales = kwargs["tensor_dict"][node.inputs[1]]

    n_in_scales_is_one = tf.equal(scales[0], 1)
    c_in_scales_is_one = tf.logical_or(tf.equal(scales[1], 1),
                                       tf.equal(scales[3], 1))
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
        y = tf.image.resize(x_t, size=new_h_w_shape, method=mode)
        y_t = tf.transpose(y, perm=[0, 3, 1, 2])
        return y_t

      def process_NHWC_format(x):
        y = tf.image.resize(x, size=new_h_w_shape, method=mode)
        return y

      output = tf.cond(x_in_NCHW_format, lambda: process_NCHW_format(x),
                       lambda: process_NHWC_format(x))

      return [output]

  @classmethod
  def version_11(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]
    x_shape = tf_shape(x)
    roi = tensor_dict[node.inputs[1]]
    scales = tensor_dict[node.inputs[2]]
    sizes = tensor_dict[node.inputs[3]] if len(
        node.inputs) == 4 else tf.constant([], dtype=tf.int64)
    coordinate_transformation_mode = node.attrs.get(
        "coordinate_transformation_mode", "half_pixel")
    extrapolation_value = node.attrs.get("extrapolation_value", 0.0)
    mode = node.attrs.get("mode", "nearest")

    param = scales if len(node.inputs) == 3 else sizes
    n_in_param_is_one = tf.equal(param[0], 1)
    c_in_param_is_one = tf.logical_or(tf.equal(param[1], 1),
                                      tf.equal(param[3], 1))
    assert_n_c_in_param_are_ones = tf.Assert(
        tf.logical_and(n_in_param_is_one, c_in_param_is_one), [param])

    with tf.control_dependencies([assert_n_c_in_param_are_ones]):
      if mode.lower() == "linear":
        mode = tf.image.ResizeMethod.BILINEAR
        tf_resize = tf.compat.v1.image.resize_bilinear
      elif mode.lower() == "cubic":
        mode = tf.image.ResizeMethod.BICUBIC
        tf_resize = tf.compat.v1.image.resize_bicubic
      else:
        mode = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        tf_resize = tf.compat.v1.image.resize_nearest_neighbor

      x_in_NCHW_format = tf.equal(param[1], 1)

      if len(node.inputs) == 3:  # only scales is defined
        h_w_scale = tf.where(x_in_NCHW_format, scales[2:], scales[1:3])
        h_w_shape = tf.where(x_in_NCHW_format, x_shape[2:], x_shape[1:3])
        new_size = tf.cast(h_w_scale * tf.cast(h_w_shape, scales.dtype),
                           tf.int32)
      else:  # sizes is defined
        # The number of elements of 'sizes' should be the same as the rank of input 'X'
        sizes.set_shape(x_shape.shape)
        new_size = tf.cast(tf.where(x_in_NCHW_format, sizes[2:], sizes[1:3]),
                           tf.int32)
      # Tensorflow require the shape of "size" in the "tf.image.resize" must be known at
      # graph creation time. However in the dynamic shape situation, the shape of "new_size"
      # will be "None", the actual shape can only be determine at runtime. But we know
      # "new_size" should always contain [h, w], therefore the shape must be 2.
      new_size.set_shape([2])

      def get_NCHW_boxes():
        indices = []
        x_rank = len(x.get_shape())
        for i in range(2, x_rank):
          indices.insert(i - 2, i)
          indices.insert(i, i + x_rank)
        return tf.expand_dims(tf.gather(roi, indices, axis=0), 0)

      def get_NHWC_boxes():
        indices = []
        x_rank = len(x.get_shape())
        for i in range(1, x_rank - 1):
          indices.insert(i - 1, i)
          indices.insert(i + 1, i + x_rank)
        return tf.expand_dims(tf.gather(roi, indices, axis=0), 0)

      box_indices = tf.cast(tf.range(0, x_shape[0]), dtype=tf.int32)

      def process_NCHW_format():
        x_t = tf.transpose(x, perm=[0, 2, 3, 1])
        if coordinate_transformation_mode == "tf_crop_and_resize":
          boxes = get_NCHW_boxes()
          y = tf.image.crop_and_resize(x_t, boxes, box_indices, new_size, mode,
                                       extrapolation_value)
        elif coordinate_transformation_mode == "align_corners":
          y = tf_resize(x_t,
                        size=new_size,
                        align_corners=True,
                        half_pixel_centers=False)
        elif coordinate_transformation_mode == "asymmetric":
          y = tf_resize(x_t,
                        size=new_size,
                        align_corners=False,
                        half_pixel_centers=False)
        else:  # half_pixel or tf_half_pixel_for_nn
          y = tf.image.resize(x_t, size=new_size, method=mode)
        return tf.transpose(y, perm=[0, 3, 1, 2])

      def process_NHWC_format():
        if coordinate_transformation_mode == "tf_crop_and_resize":
          boxes = get_NHWC_boxes()
          return tf.image.crop_and_resize(x, boxes, box_indices, new_size, mode,
                                          extrapolation_value)
        elif coordinate_transformation_mode == "align_corners":
          return tf_resize(x,
                           size=new_size,
                           align_corners=True,
                           half_pixel_centers=False)
        elif coordinate_transformation_mode == "asymmetric":
          return tf_resize(x,
                           size=new_size,
                           align_corners=False,
                           half_pixel_centers=False)
        else:  # half_pixel or tf_half_pixel_for_nn
          return tf.image.resize(x, size=new_size, method=mode)

      output = tf.cond(x_in_NCHW_format, process_NCHW_format,
                       process_NHWC_format)

      return [output]
