from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
from onnx import helper
from onnx.onnx_pb2 import TensorProto

MODEL_PATH = "../../onnx_models/"

def _test_nn(net, output):
  model = onnx.load(MODEL_PATH + net + "/model.pb")
  tf_rep = prepare(model)
  for i in range(3):
    sample = np.load(MODEL_PATH + net + "/test_data_{}.npz".format(str(i)), encoding='bytes')

    inputs = list(sample['inputs'])
    outputs = list(sample['outputs'])

    my_out = tf_rep.run(inputs)
    np.testing.assert_allclose(outputs[0], my_out[output], rtol=1e-3)

class TestLargeModel(unittest.TestCase):

  def test_squeezenet(self):
    _test_nn("squeezenet", "softmaxout_1")

  def test_vgg16(self):
    _test_nn("vgg16", "gpu_0/softmax_1")

  def test_vgg19(self):
    _test_nn("vgg19", "prob_1")

  def test_bvlc_alexnet(self):
    _test_nn("bvlc_alexnet", "prob_1")

  def test_shuffle_net(self):
    return
    _test_nn("shufflenet", "gpu_0/softmax_1")

  def test_dense_net(self):
    _test_nn("densenet121", "fc6_1")

  def test_resnet50(self):
    _test_nn("resnet50", "gpu_0/softmax_1")

  def test_inception_v1(self):
    _test_nn("inception_v1", "prob_1")

  def test_inception_v2(self):
    _test_nn("inception_v2", "prob_1")

if __name__ == '__main__':
  unittest.main()
