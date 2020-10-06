import inspect
import os
import subprocess
import unittest

import onnx
from onnx.backend.test.runner import Runner
from onnx.backend.test.case.model import TestCase

from onnx_tf.backend import TensorflowBackend
from onnx_tf.common import IS_PYTHON3
from onnx_tf.common.legacy import legacy_onnx_pre_ver

_ONNX_MODELS = [(
    "mobilenetv2-1.0",
    "https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz"
)]


class TestCli(unittest.TestCase):

  @staticmethod
  def prepare_model(model_name, url):
    if legacy_onnx_pre_ver(1, 5, 0):
      prepare_model_data = Runner._prepare_model_data
    else:
      prepare_model_data = Runner.prepare_model_data
    if IS_PYTHON3:
      params = list(
          inspect.signature(prepare_model_data).parameters.keys())
    else:
      params = inspect.getargspec(prepare_model_data).args
    runner_class = Runner
    if params[0] == "self":
      runner_class = Runner(TensorflowBackend)
      if legacy_onnx_pre_ver(1, 5, 0):
        prepare_model_data = runner_class._prepare_model_data
      else:
        prepare_model_data = runner_class.prepare_model_data
    if legacy_onnx_pre_ver(1, 4, 0):
      tc = TestCase(
          name="test_{}".format(model_name),
          model_name=model_name,
          url=url,
          model_dir=None,
          model=None,
          data_sets=None,
          kind='real')
    else:
      tc = TestCase(
          name="test_{}".format(model_name),
          model_name=model_name,
          url=url,
          model_dir=None,
          model=None,
          data_sets=None,
          kind='real',
          rtol=1e-3,
          atol=1e-7)
    return prepare_model_data(model_test=tc)

  def test_convert_to_tf(self):
    if legacy_onnx_pre_ver(1, 2, 1):
      raise unittest.SkipTest(
          "The current version of ONNX uses dead model link.")
    for model_name, url in _ONNX_MODELS:
      model_dir = self.prepare_model(model_name, url)
      subprocess.check_call([
          "onnx-tf",
          "convert",
          "-i",
          os.path.join(model_dir, '{}.onnx'.format(model_name)),
          "-o",
          os.path.join(model_dir, '{}.pb'.format(model_name)),
      ])


if __name__ == '__main__':
  unittest.main()
