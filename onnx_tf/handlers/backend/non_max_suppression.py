import tensorflow as tf

from onnx_tf.common import get_variable_name
from onnx_tf.common.tf_helper import tf_shape
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("NonMaxSuppression")
class NonMaxSuppression(BackendHandler):
  var_name = 'result'

  @classmethod
  def get_req_vars_template(cls, node, init_dict):
    """ Get required variables template, which is a
    dictionary of variable names with initial value and
    shape.
    :param node: OnnxNode object.
    :param init_dict: initializer dictionary of the graph.
    :return: template dictionary.
    """
    return {
        cls.var_name: [
            tf.constant([[0, 0, 0]], dtype=tf.int64),
            tf.TensorShape([None, 3])
        ]
    }

  @classmethod
  def _common(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    boxes = tensor_dict[node.inputs[0]]
    scores = tensor_dict[node.inputs[1]]
    # in ONNX spec max_output_boxes_per_class need to be in int64 but
    # max_output_boxes for tf.image.non_max_suppression must be in tf.int32
    # therefore need to cast this input to tf.int32
    max_output_boxes_per_class = tf.cast(
        tensor_dict[node.inputs[2]],
        tf.int32) if (len(node.inputs) > 2 and
                      node.inputs[2] != "") else tf.constant(0, tf.int32)
    # make sure max_output_boxes_per_class is a scalar not a 1-D 1 element tensor
    max_output_boxes_per_class = tf.squeeze(max_output_boxes_per_class) if len(
        max_output_boxes_per_class.shape) == 1 else max_output_boxes_per_class
    iou_threshold = tensor_dict[node.inputs[3]] if (
        len(node.inputs) > 3 and node.inputs[3] != "") else tf.constant(
            0, tf.float32)
    # make sure iou_threshold is a scalar not a 1-D 1 element tensor
    iou_threshold = tf.squeeze(iou_threshold) if len(
        iou_threshold.shape) == 1 else iou_threshold
    score_threshold = tensor_dict[node.inputs[4]] if (
        len(node.inputs) > 4 and node.inputs[4] != "") else tf.constant(
            float('-inf'))
    # make sure score_threshold is a scalar not a 1-D 1 element tensor
    score_threshold = tf.squeeze(score_threshold) if len(
        score_threshold.shape) == 1 else score_threshold
    center_point_box = node.attrs.get("center_point_box", 0)

    if center_point_box == 1:
      boxes_t = tf.transpose(boxes, perm=[0, 2, 1])
      x_centers = tf.slice(boxes_t, [0, 0, 0], [-1, 1, -1])
      y_centers = tf.slice(boxes_t, [0, 1, 0], [-1, 1, -1])
      widths = tf.slice(boxes_t, [0, 2, 0], [-1, 1, -1])
      heights = tf.slice(boxes_t, [0, 3, 0], [-1, 1, -1])
      y1 = tf.subtract(y_centers, tf.divide(heights, 2))
      x1 = tf.subtract(x_centers, tf.divide(widths, 2))
      y2 = tf.add(y_centers, tf.divide(heights, 2))
      x2 = tf.add(x_centers, tf.divide(widths, 2))
      boxes_t = tf.concat([y1, x1, y2, x2], 1)
      boxes = tf.transpose(boxes_t, perm=[0, 2, 1])

    def create_nodes(boxes, scores, max_output_boxes_per_class, iou_threshold,
                     score_threshold, result):
      # get number of batches in boxes
      num_batches = tf_shape(boxes)[0]
      for batch_i in tf.range(num_batches):
        # get boxes in batch_i only
        tf_boxes = tf.squeeze(tf.gather(boxes, [batch_i]), axis=0)
        # get scores of all classes in batch_i only
        batch_i_scores = tf.squeeze(tf.gather(scores, [batch_i]), axis=0)
        # get number of classess in batch_i only
        num_classes = tf_shape(batch_i_scores)[0]
        for class_j in tf.range(num_classes):
          # get scores in class_j for batch_i only
          tf_scores = tf.squeeze(tf.gather(batch_i_scores, [class_j]), axis=0)
          # get the selected boxes indices
          selected_indices = tf.image.non_max_suppression(
              tf_boxes, tf_scores, max_output_boxes_per_class, iou_threshold,
              score_threshold)
          # add batch and class information into the indices
          output = tf.transpose([tf.cast(selected_indices, dtype=tf.int64)])
          paddings = tf.constant([[0, 0], [1, 0]])
          output = tf.pad(output,
                          paddings,
                          constant_values=tf.cast(class_j, dtype=tf.int64))
          output = tf.pad(output,
                          paddings,
                          constant_values=tf.cast(batch_i, dtype=tf.int64))
          # tf.function will auto convert "result" from variable to placeholder
          # therefore don't need to use assign here
          result = output if tf.equal(batch_i, 0) and tf.equal(
              class_j, 0) else tf.concat([result, output], 0)

      return result

    result = tensor_dict[get_variable_name(node, cls.var_name)]
    return [
        create_nodes(boxes, scores, max_output_boxes_per_class, iou_threshold,
                     score_threshold, result)
    ]

  @classmethod
  def version_10(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)
