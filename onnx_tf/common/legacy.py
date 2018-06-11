from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx

def get_onnx_version():
	return map(int, onnx.version.version.split("."))

# Returns whether onnx version is prior to 1.2.
def legacy_onnx_pre_1_2():
	major, minor, revision = get_onnx_version()
	return major == 1 and minor < 2

# Returns whether the opset version accompanying the
# onnx installation is prior to 6.
def legacy_opset_pre_6():
	return onnx.defs.onnx_opset_version() < 6