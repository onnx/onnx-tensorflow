from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


class Handler(object):
  TF_OP = []
  ONNX_OP = None

  DOMAIN = ""
  VERSION = 0
  SINCE_VERSION = 0
