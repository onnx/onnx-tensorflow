import re

from . import op_name_to_lower
from onnx_tf.handlers.frontend import *  # noqa
from onnx_tf.handlers.frontend_handler import FrontendHandler


def __get_all_subclasses(clazz, except_regex=None):
  all_subclasses = set(clazz.__subclasses__()).union([
      s for c in clazz.__subclasses__()
      for s in __get_all_subclasses(c, except_regex=except_regex)
  ])
  if except_regex is not None:
    all_subclasses = set(
        filter(lambda c: re.match(except_regex, c.__name__) is None,
               all_subclasses))
  return all_subclasses


def get_all_frontend_handlers():
  handlers = {}
  for handler in __get_all_subclasses(
      FrontendHandler, except_regex=r'.*Common$'):
    domain = getattr(handler, "DOMAIN")
    for tf_op in handler.get_tf_op():
      handlers.setdefault(domain, {})[tf_op] = handler
  return handlers


def get_frontend_coverage():
  tf_coverage = {}
  onnx_coverage = {}
  for handler in __get_all_subclasses(
      FrontendHandler, except_regex=r'.*Common$'):
    versions = handler.get_versions()
    domain = getattr(handler, "DOMAIN")
    for tf_op in handler.get_tf_op():
      tf_coverage.setdefault(domain, {})[op_name_to_lower(tf_op)] = versions
    onnx_coverage.setdefault(domain, {})[handler.get_onnx_op()] = versions
  return onnx_coverage, tf_coverage
