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

  _cls_ver_handle = {}
  _cls_versions = {}

  @classmethod
  def version(cls, ver):
    def decorator(func):
      class_name = func.__qualname__.split(".")[0]
      cls_ver_handle = cls._cls_ver_handle.setdefault(class_name, {})
      cls_ver_handle[ver] = func
      cls_versions = cls._cls_versions.setdefault(class_name, [])
      cls_versions.append(ver)
      return func
    return decorator

  @classmethod
  def get_versions(cls):
    return cls._cls_versions.get(cls.__name__, [])

  @classmethod
  def get_ver_handle(cls, ver):
    return cls._cls_ver_handle.get(cls.__name__, {}).get(ver, None)
