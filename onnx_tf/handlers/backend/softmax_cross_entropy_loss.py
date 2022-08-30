import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("SoftmaxCrossEntropyLoss")
@tf_func(tf.nn.sparse_softmax_cross_entropy_with_logits)
class SoftmaxCrossEntropyLoss(BackendHandler):
    @classmethod
    def _common(cls, node, **kwargs):
        logits = kwargs["tensor_dict"][node.inputs[0]]
        labels = kwargs["tensor_dict"][node.inputs[1]]

        labels_shape = tf.shape(labels)
        if labels_shape.shape[0] > 1:
            raise NotImplementedError(
                "SoftmaxCrossEntropyLoss support is limited to rank 1 label tensors."
                .format(spatial_size))

        return [
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        ]

    @classmethod
    def version_12(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)
