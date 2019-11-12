from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from .unpool_mixin import UnpoolMixin


@onnx_op("MaxUnpool")
class MaxUnpool(UnpoolMixin, BackendHandler):

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls.max_unpool(node, kwargs["tensor_dict"])

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls.max_unpool(node, kwargs["tensor_dict"])
