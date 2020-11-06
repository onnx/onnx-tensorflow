import tensorflow as tf

from onnx_tf.common import logger
from onnx_tf.common import get_data_format
from onnx_tf.common import get_perm_from_formats

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import partial_support
from onnx_tf.handlers.handler import ps_description
from onnx_tf.handlers.handler import onnx_op


def crop_and_resize(image,
                    boxes,
                    box_ind,
                    crop_size,
                    sampling_ratio,
                    adaptive_ratio=False,
                    pad_border=False):

    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)

    # TF's crop_and_resize produces zeros on border
    if pad_border:
        image = tf.pad(image, [[0, 0], [1, 1], [1, 1], [0, 0]],
                       mode='SYMMETRIC')
        boxes = boxes + 1

    def transform_fpcoor_for_tf(boxes, image_shape, crop_size, sampling_ratio,
                                adaptive_ratio):

        x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

        if not adaptive_ratio:
            crop_shape = (crop_size[0] * sampling_ratio,
                          crop_size[1] * sampling_ratio)
            spacing_w = (x1 - x0) / tf.cast(crop_shape[1], dtype=tf.float32)
            spacing_h = (y1 - y0) / tf.cast(crop_shape[0], dtype=tf.float32)

            nx0 = (x0 + spacing_w / 2) / tf.cast(image_shape[1] - 1,
                                                 dtype=tf.float32)
            ny0 = (y0 + spacing_h / 2) / tf.cast(image_shape[0] - 1,
                                                 dtype=tf.float32)

            nw = spacing_w * tf.cast(crop_shape[1] - 1,
                                     dtype=tf.float32) / tf.cast(
                                         image_shape[1] - 1, dtype=tf.float32)
            nh = spacing_h * tf.cast(crop_shape[0] - 1,
                                     dtype=tf.float32) / tf.cast(
                                         image_shape[0] - 1, dtype=tf.float32)
        else:

            # TODO: find a better method when adaptive_ratio=True
            roi_width = x1 - x0
            roi_height = y1 - y0
            nx0 = x0 / tf.cast(image_shape[1] - 1, dtype=tf.float32)
            ny0 = y0 / tf.cast(image_shape[0] - 1, dtype=tf.float32)
            nw = (roi_width - 1) / tf.cast(image_shape[1] - 1,
                                           dtype=tf.float32)
            nh = (roi_height - 1) / tf.cast(image_shape[0] - 1,
                                            dtype=tf.float32)

        return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

    image_shape = tf.shape(image)[1:3]
    boxes = transform_fpcoor_for_tf(boxes, image_shape, crop_size,
                                    sampling_ratio, adaptive_ratio)
    ret = tf.image.crop_and_resize(
        image,
        boxes,
        tf.cast(box_ind, dtype=tf.int32),
        crop_size=(crop_size[0] * sampling_ratio,
                   crop_size[1] * sampling_ratio),
    )
    return ret


@onnx_op("RoiAlign")
@partial_support(True)
@ps_description("sampling_ratio <= 0 is not fully supported.")
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
        adaptive_ratio = False
        if sampling_ratio <= 0:
            sampling_ratio = int((output_height + output_width) / 2)
            adaptive_ratio = True
            logger.warning("Do not fully support sampling_ratio <= 0.")

        boxes = boxes * spatial_scale

        feat_rank = len(feat.shape)
        storage_format, _ = get_data_format(feat_rank)
        need_trans = storage_format.startswith("NC")
        if need_trans:
            compute_format = "N" + storage_format[2:] + "C"
            feat = tf.transpose(feat,
                                perm=get_perm_from_formats(
                                    storage_format, compute_format))

        ret = crop_and_resize(feat,
                              boxes,
                              tf.cast(indx, tf.int32),
                              (output_height, output_width),
                              sampling_ratio,
                              adaptive_ratio=adaptive_ratio)
        ret = tf.nn.avg_pool(ret, [1, sampling_ratio, sampling_ratio, 1],
                             [1, sampling_ratio, sampling_ratio, 1],
                             padding='SAME',
                             data_format='NHWC')
        ret = tf.transpose(ret, perm=(0, 3, 1, 2))
        return [ret]

    @classmethod
    def version_10(cls, node, **kwargs):
        return cls._common(node, **kwargs)
