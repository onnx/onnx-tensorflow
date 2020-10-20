import tensorflow as tf

from onnx_tf.common import get_data_format
from onnx_tf.common import get_perm_from_formats

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


def crop_and_resize(image, boxes, box_ind, crop_size, pad_border=True):
    """
    Aligned version of tf.image.crop_and_resize, following our definition of floating point boxes.
    Args:
        image: NHWC
        boxes: nx4, x1y1x2y2
        box_ind: (n,)
        crop_size (int):
    Returns:
        n,C,size,size
    """

    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)

    # TF's crop_and_resize produces zeros on border
    if pad_border:
        # this can be quite slow
        image = tf.pad(image, [[0, 0], [1, 1], [1, 1], [0, 0]],
                       mode='SYMMETRIC')
        boxes = boxes + 1

    def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
        """
        The way tf.image.crop_and_resize works (with normalized box):
        Initial point (the value of output[0]): x0_box * (W_img - 1)
        Spacing: w_box * (W_img - 1) / (W_crop - 1)
        Use the above grid to bilinear sample.
        However, what we want is (with fpcoor box):
        Spacing: w_box / W_crop
        Initial point: x0_box + spacing/2
        This function transform fpcoor boxes to a format to be used by tf.image.crop_and_resize
        Returns:
            y1x1y2x2
        """
        x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

        spacing_w = (x1 - x0) / tf.to_float(crop_shape[1])
        spacing_h = (y1 - y0) / tf.to_float(crop_shape[0])

        nx0 = (x0 + spacing_w / 2) / tf.to_float(image_shape[1] - 1)
        ny0 = (y0 + spacing_h / 2) / tf.to_float(image_shape[0] - 1)

        nw = spacing_w * tf.to_float(crop_shape[1] -
                                     1) / tf.to_float(image_shape[1] - 1)
        nh = spacing_h * tf.to_float(crop_shape[0] -
                                     1) / tf.to_float(image_shape[0] - 1)

        return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

    image_shape = tf.shape(image)[1:3]
    boxes = transform_fpcoor_for_tf(boxes, image_shape, crop_size)
    ret = tf.compat.v1.image.crop_and_resize(image,
                                             boxes,
                                             tf.to_int32(box_ind),
                                             crop_size=crop_size)
    return ret


@onnx_op("RoiAlign")
class RoiAlign(BackendHandler):
    @classmethod
    def _common(cls, node, **kwargs):
        tensor_dict = kwargs['tensor_dict']
        feat = tensor_dict[node.inputs[0]]
        boxes = tensor_dict[node.inputs[1]]
        indx = tensor_dict[node.inputs[2]]
        output_height = node.attrs['output_height']
        output_width = node.attrs['output_width']
        sampling_ratio = node.attrs['sampling_ratio']
        spatial_scale = node.attrs['spatial_scale']
        if sampling_ratio <= 0:
            sampling_ratio = 2

        boxes = boxes * spatial_scale

        feat_rank = len(feat.shape)
        storage_format, _ = get_data_format(feat_rank)
        need_trans = storage_format.startswith("NC")
        if need_trans:
            compute_format = "N" + storage_format[2:] + "C"
            feat = tf.transpose(feat,
                                perm=get_perm_from_formats(
                                    storage_format, compute_format))

        ret = crop_and_resize(
            feat, boxes, tf.cast(indx, tf.int32),
            (output_height * sampling_ratio, output_width * sampling_ratio))
        ret = tf.nn.avg_pool(ret, [1, sampling_ratio, sampling_ratio, 1],
                             [1, sampling_ratio, sampling_ratio, 1],
                             padding='SAME',
                             data_format='NHWC')
        ret = tf.transpose(ret, perm=(0, 3, 1, 2))
        return [ret]

    @classmethod
    def version_10(cls, node, **kwargs):
        return cls._common(node, **kwargs)
