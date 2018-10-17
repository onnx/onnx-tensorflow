import warnings

from onnx import defs

from onnx_tf.handlers.backend import *  # noqa
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.frontend import *  # noqa
from onnx_tf.handlers.frontend_handler import FrontendHandler


def get_all_frontend_handlers(opset_dict):
  """ Get a dict of all frontend handler classes.
  e.g. {'domain': {'Abs': Abs handler class}, ...}, }.

  :param opset_dict: A dict of opset. e.g. {'domain': version, ...}
  :return: Dict.
  """
  handlers = {}
  for handler in FrontendHandler.__subclasses__():
    handler.check_cls()

    domain = handler.DOMAIN
    version = opset_dict[domain]
    handler.VERSION = version

    since_version = 1
    if handler.ONNX_OP and defs.has(handler.ONNX_OP, domain=handler.DOMAIN):
      since_version = defs.get_schema(
          handler.ONNX_OP, domain=handler.DOMAIN,
          max_inclusive_version=version).since_version
    else:
      warnings.warn("Unknown op {} in domain `{}`. "
                    "Can't check specification by ONNX. "
                    "Please set should_check flag to False "
                    "when call make_node method in handler.".format(
                        handler.ONNX_OP or "Undefined", handler.DOMAIN or
                        "ai.onnx"))
    handler.SINCE_VERSION = since_version

    for tf_op in handler.TF_OP:
      handlers.setdefault(domain, {})[tf_op] = handler
  return handlers


def get_all_backend_handlers(opset_dict):
  """ Get a dict of all backend handler classes.
  e.g. {'domain': {'Abs': Abs handler class}, ...}, }.

  :param opset_dict: A dict of opset. e.g. {'domain': version, ...}
  :return: Dict.
  """
  handlers = {}
  for handler in BackendHandler.__subclasses__():
    handler.check_cls()

    domain = handler.DOMAIN
    version = opset_dict[domain]
    handler.VERSION = version

    since_version = 1
    if defs.has(handler.ONNX_OP, domain=handler.DOMAIN):
      try:
        since_version = defs.get_schema(
            handler.ONNX_OP,
            domain=handler.DOMAIN,
            max_inclusive_version=version).since_version
      except RuntimeError:
        warnings.warn("Fail to get since_version of {} in domain `{}` "
                      "with max_inclusive_version={}. Set to 1.".format(
                          handler.ONNX_OP, handler.DOMAIN, version))
    else:
      warnings.warn("Unknown op {} in domain `{}`.".format(
          handler.ONNX_OP, handler.DOMAIN or "ai.onnx"))
    handler.SINCE_VERSION = since_version
    handlers.setdefault(domain, {})[handler.ONNX_OP] = handler
  return handlers


def get_frontend_coverage():
  """ Get frontend coverage for document.

  :return: onnx_coverage: e.g. {'domain': {'ONNX_OP': [versions], ...}, ...}
  tf_coverage: e.g. {'domain': {'TF_OP': [versions], ...}, ...}
  """

  tf_coverage = {}
  onnx_coverage = {}
  for handler in FrontendHandler.__subclasses__():
    handler.check_cls()
    versions = handler.get_versions()
    domain = handler.DOMAIN
    for tf_op in handler.TF_OP:
      _update_coverage(tf_coverage, domain, tf_op, versions)
    if handler.ONNX_OP:
      _update_coverage(onnx_coverage, domain, handler.ONNX_OP, versions)
  return onnx_coverage, tf_coverage


def get_backend_coverage():
  """ Get backend coverage for document.

  :return: onnx_coverage: e.g. {'domain': {'ONNX_OP': [versions], ...}, ...}
  """

  onnx_coverage = {}
  for handler in BackendHandler.__subclasses__():
    handler.check_cls()

    versions = handler.get_versions()
    domain = handler.DOMAIN
    _update_coverage(onnx_coverage, domain, handler.ONNX_OP, versions)
  return onnx_coverage


def _update_coverage(coverage, domain, key, versions):
  domain_coverage = coverage.setdefault(domain, {})
  vers = domain_coverage.get(key, [])
  vers.extend(versions)
  domain_coverage[key] = sorted(list(set(vers)))
