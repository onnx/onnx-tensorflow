import os
import subprocess

from onnx.backend.test.runner import Runner
from onnx.backend.test.case.model import TestCase
import unittest

_ONNX_MODELS = [(
    "mobilenetv2-1.0",
    "https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz"
)]


class TestCli(unittest.TestCase):

  @staticmethod
  def prepare_model(model_name, url):
    test_case = TestCase(
        name="test_{}".format(model_name),
        model_name=model_name,
        url=url,
        model_dir=None,
        model=None,
        data_sets=None,
        kind='real',
    )
    return Runner._prepare_model_data(test_case)

  def test_convert_to_tf(self):
    for model_name, url in _ONNX_MODELS:
      model_dir = self.prepare_model(model_name, url)
      subprocess.check_call([
          "onnx-tf",
          "convert",
          "-t",
          "tf",
          "-i",
          os.path.join(model_dir, '{}.onnx'.format(model_name)),
          "-o",
          os.path.join(model_dir, '{}.pb'.format(model_name)),
      ])


if __name__ == '__main__':
  unittest.main()
