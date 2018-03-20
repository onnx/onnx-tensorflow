"""Backend for running ONNX on Tensorflow

To run this, you will need to have Tensorflow installed as well.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from onnx_tf.backend import TensorflowBackendBase
import tensorflow as tf

class TensorflowBackend(TensorflowBackendBase):
  """ Tensorflow Backend for ONNX
  """
