import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("NonMaxSuppression")
class NonMaxSuppression(BackendHandler):

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

    # get number of batches in boxes
    num_batches = boxes.shape[0]
    for batch_i in range(num_batches):
      # get boxes in batch_i only
      tf_boxes = tf.squeeze(tf.gather(boxes, [batch_i]), axis=0)
      # get scores of all classes in batch_i only
      batch_i_scores = tf.squeeze(tf.gather(scores, [batch_i]), axis=0)
      # get number of classess in batch_i only
      num_classes = batch_i_scores.shape[0]
      for class_j in range(num_classes):
        # get scores in class_j for batch_i only
        tf_scores = tf.squeeze(tf.gather(batch_i_scores, [class_j]), axis=0)
        # get the selected boxes indices
        selected_indices = tf.image.non_max_suppression(
            tf_boxes, tf_scores, max_output_boxes_per_class, iou_threshold,
            score_threshold)
        # add batch and class information into the indices
        output = tf.transpose([tf.cast(selected_indices, dtype=tf.int64)])
        paddings = tf.constant([[0, 0], [1, 0]])
        output = tf.pad(output, paddings, constant_values=class_j)
        output = tf.pad(output, paddings, constant_values=batch_i)
        result = tf.concat([result, output],
                           0) if 'result' in locals() else output

    return [result]

  @classmethod
  def version_10(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)
