import numpy as np
import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("EyeLike")
class EyeLike(BackendHandler):

  @classmethod
  def version_9(cls, node, **kwargs):

    inp = kwargs["tensor_dict"][node.inputs[0]]
    dtype = node.attrs.pop("dtype", inp.dtype)
    offset = node.attrs.pop("k", 0)

    # If the shape of input is static, then the handler
    # can use python code to calculate the eye shape and
    # paddings for pad in the graph generating phase.
    # Then the graph will only contains two nodes:
    # tf.eye and tf.pad
    if inp.shape.is_fully_defined():

      shape = inp.shape.as_list()
      # calculate upper and lower bound of max eye shape
      max_eye_shape_ub = shape[1] if offset > 0 else shape[0]
      max_eye_shape_lb = shape[0] if offset > 0 else shape[1]
      # adjust offset base on max_eye_shape_ub
      offset = max_eye_shape_ub * np.sign(offset) if abs(
          offset) > max_eye_shape_ub else offset
      abs_offset = abs(offset)
      eye_shape = min(max_eye_shape_ub - abs_offset, max_eye_shape_lb)
      tensor = tf.eye(eye_shape, num_columns=eye_shape, dtype=dtype)
      if offset > 0:
        tb_paddings = [0, shape[0] - eye_shape]
        lr_paddings = [offset, shape[1] - offset - eye_shape]
      else:
        tb_paddings = [abs_offset, shape[0] - abs_offset - eye_shape]
        lr_paddings = [0, shape[1] - eye_shape]
      paddings = tf.constant([tb_paddings, lr_paddings], dtype=tf.int32)
      return [
          cls.make_tensor_from_onnx_node(node,
                                         tf_func=tf.pad,
                                         inputs=[tensor, paddings],
                                         **kwargs)
      ]

    # if the input shape is not defined yet during the graph generaring
    # phase, then need to perform the eye shape and paddings calculation
    # during the execution phase. Therefore the graph will be bigger.
    # Using tf.funtion to let Tensorflow auto generate a graph out of
    # our python code would be the best option in this case to get the
    # smallest graph possible
    else:
      def create_nodes(inp, offset):
        shape = tf.shape(inp, out_type=tf.int32)
        # calculate upper and lower bound of max eye shape
        max_eye_shape_ub = shape[1] if offset > 0 else shape[0]
        max_eye_shape_lb = shape[0] if offset > 0 else shape[1]
        # adjust offset base on max_eye_shape_ub
        offset = max_eye_shape_ub * np.sign(offset) if abs(
            offset) > max_eye_shape_ub else offset
        abs_offset = abs(offset)
        eye_shape = tf.minimum(max_eye_shape_ub - abs_offset, max_eye_shape_lb)
        tensor = tf.eye(eye_shape, num_columns=eye_shape, dtype=dtype)
        if offset > 0:
          tb_paddings = [0, shape[0] - eye_shape]
          lr_paddings = [offset, shape[1] - offset - eye_shape]
        else:
          tb_paddings = [abs_offset, shape[0] - abs_offset - eye_shape]
          lr_paddings = [0, shape[1] - eye_shape]
        paddings = [tb_paddings, lr_paddings]
        return tensor, paddings

      tensor, paddings = create_nodes(inp, offset)
      return [
          cls.make_tensor_from_onnx_node(node,
                                         tf_func=tf.pad,
                                         inputs=[tensor, paddings],
                                         **kwargs)
      ]
