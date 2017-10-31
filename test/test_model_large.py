from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import tensorflow as tf
import onnx
from onnx_tf.backend import run_model, prepare
from onnx import helper
from onnx.onnx_pb2 import TensorProto

class TestNode(unittest.TestCase):
  MODEL_PATH = "../../onnx_models/"

  def test_squeezenet(self):
    for i in range(3):
      model = onnx.load(self.MODEL_PATH + "squeezenet/model.pb")
      sample = np.load(self.MODEL_PATH + "squeezenet/test_data_{}.npz".format(str(i)), encoding='bytes')

      inputs = list(sample['inputs'])
      outputs = list(sample['outputs'])

      tf_rep = prepare(model)
      my_out = tf_rep.run(inputs)
      np.testing.assert_almost_equal(outputs[0], my_out['softmaxout_1'], decimal=5)

if __name__ == '__main__':
  unittest.main()
