import re
import warnings

from onnx import defs

from . import op_name_to_lower
from onnx_tf.handlers.frontend import *  # noqa
from onnx_tf.handlers.frontend_handler import FrontendHandler


def get_all_frontend_handlers(opset_dict):
  handlers = {}
  for handler in FrontendHandler.__subclasses__():
    domain = getattr(handler, "DOMAIN")
    version = opset_dict[domain]
    handler.VERSION = version

    since_version = 1
    if defs.has(handler.get_onnx_op(), domain=handler.DOMAIN):
      since_version = defs.get_schema(
          handler.get_onnx_op(),
          domain=handler.DOMAIN,
          max_inclusive_version=version).since_version
    else:
      warnings.warn("Unknown op {} in domain `{}`. "
                    "Can't check specification by ONNX. "
                    "Please set should_check flag to False "
                    "when call make_node method in handler.".format(
                        handler.get_onnx_op(), handler.DOMAIN or "ai.onnx"))
    handler.SINCE_VERSION = since_version

    for tf_op in handler.get_tf_op():
      handlers.setdefault(domain, {})[tf_op] = handler
  return handlers


def get_frontend_coverage():
  tf_coverage = {}
  onnx_coverage = {}
  for handler in FrontendHandler.__subclasses__():
    versions = handler.get_versions()
    domain = getattr(handler, "DOMAIN")
    for tf_op in handler.get_tf_op():
      tf_coverage.setdefault(domain, {})[op_name_to_lower(tf_op)] = versions
    onnx_coverage.setdefault(domain, {})[handler.get_onnx_op()] = versions
  return onnx_coverage, tf_coverage
