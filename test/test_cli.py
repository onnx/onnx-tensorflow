import os
import subprocess

from onnx.backend.test.runner import Runner
from onnx.backend.test.case.model import TestCase
import unittest

_MODELS = [(
    "mobilenetv2-1.0",
    "https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz"
)]


class TestCli(unittest.TestCase):

  def prepare_model(self, model_name, url):
    return Runner._prepare_model_data(
        TestCase(
            name="test_{}".format(model_name),
            model_name=model_name,
            url=url,
            model_dir=None,
            model=None,
            data_sets=None,
            kind='real',
        ))

  def test_cli(self):
    for model_name, url in _MODELS:
      model_dir = self.prepare_model(model_name, url)
      subprocess.check_call([
          "onnx-tf",
          "convert",
          "-t",
          "onnx",
          "-i",
          os.path.join(model_dir, '{}.onnx'.format(model_name)),
          "-o",
          os.path.join(model_dir, '{}.pb'.format(model_name)),
      ])


if __name__ == '__main__':
  unittest.main()
