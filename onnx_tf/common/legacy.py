from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx


def get_onnx_version():
  return tuple(map(int, onnx.version.version.split(".")))


# Returns whether onnx version is prior to major.minor.patch
def legacy_onnx_pre_ver(major=0, minor=0, patch=0):
  return get_onnx_version() < (major, minor, patch)


# Returns whether the opset version accompanying the
# onnx installation is prior to version passed.
def legacy_opset_pre_ver(version):
  return onnx.defs.onnx_opset_version() < version
