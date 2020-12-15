from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx


def get_onnx_version():
  # We will treat an onnx rc, release candidate, as a formal
  # release in the context of onnx-tf for verification purpose.
  # In formal releases, there should not be any non numeric
  # characters in onnx version.
  # The assumption is onnx release candidate version is in the 
  # pattern of 1.8.0rc, as seen in the recent 1.8, where 'rc' is
  # appended to the patch number to indicate release candidates.
  rc_index = onnx.version.version.lower().find('rc')
  onnx_version = onnx.version.version if rc_index == -1 else onnx.version.version[:rc_index]
  return tuple(map(int, onnx_version.split(".")))


# Returns whether onnx version is prior to major.minor.patch
def legacy_onnx_pre_ver(major=0, minor=0, patch=0):
  return get_onnx_version() < (major, minor, patch)


# Returns whether the opset version accompanying the
# onnx installation is prior to version passed.
def legacy_opset_pre_ver(version):
  return onnx.defs.onnx_opset_version() < version
