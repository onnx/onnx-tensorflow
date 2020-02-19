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
        tensor_dict['max_output_boxes_per_class'],
        tf.int32) if 'max_output_boxes_per_class' in tensor_dict else tf.cast(
            boxes.shape[1], tf.int32)
    iou_threshold = tensor_dict[
        'iou_threshold'] if 'iou_threshold' in tensor_dict else tf.constant(
            [0.5], tf.float32)
    score_threshold = tensor_dict[
        'score_threshold'] if 'score_threshold' in tensor_dict else tf.constant(
            [float('-inf')], tf.float32)
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

    @tf.function
    def create_nodes(boxes, scores, max_output_boxes_per_class, iou_threshold,
                     score_threshold, result):
      # get number of batches in boxes
      num_batches = tf.shape(boxes)[0]
      for batch_i in tf.range(num_batches):
        # get boxes in batch_i only
        tf_boxes = tf.squeeze(tf.gather(boxes, [batch_i]), axis=0)
        # get scores of all classes in batch_i only
        batch_i_scores = tf.squeeze(tf.gather(scores, [batch_i]), axis=0)
        # get number of classess in batch_i only
        num_classes = tf.shape(batch_i_scores)[0]
        for class_j in tf.range(num_classes):
          # get scores in class_j for batch_i only
          tf_scores = tf.squeeze(tf.gather(batch_i_scores, [class_j]), axis=0)
          # get the selected boxes indices
          selected_indices = tf.image.non_max_suppression(
              tf_boxes, tf_scores, max_output_boxes_per_class[0],
              iou_threshold[0], score_threshold[0])
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

    # Since tf.function doesn't support locals() and it require all the variables
    # are defined before use in the "for loop" before it will perform any auto
    # convertion of the python code. Therefore need to define "result" as a
    # Variable here and send it in as a parameter to "create_nodes"
    result = tf.Variable([[0, 0, 0]],
                         dtype=tf.int64,
                         shape=tf.TensorShape([None, 3]))
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
