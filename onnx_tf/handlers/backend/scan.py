import tensorflow as tf
import numpy as np
import onnx_tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx import helper
from .scan_mixin import ScanMixin


@onnx_op("Scan")
class Scan(ScanMixin, BackendHandler):


    @classmethod
    def _common(cls, node, **kwargs):
        return cls.scan(node, kwargs["tensor_dict"], kwargs.get("strict", True))


    @classmethod
    def version_8(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

