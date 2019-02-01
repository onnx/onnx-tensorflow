import warnings

from onnx import defs

from onnx_tf.handlers.backend import *  # noqa
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.frontend import *  # noqa
from onnx_tf.handlers.frontend_handler import FrontendHandler


class DomainHandlerDict(dict):

  def __init__(self, domain, unknown_message="", failed_message=""):
    self.unknown = {}
    self.failed = {}
    self._domain = domain
    self._unknown_message = unknown_message
    self._failed_message = failed_message

  def _warn(self, k):
    if k in self.unknown:
      warnings.warn(
          self._unknown_message.format(self._domain, self.unknown.pop(k)))
    if k in self.failed:
      warnings.warn(
          self._failed_message.format(self._domain, k, self.failed.pop(k)))

  def __getitem__(self, k):
    self._warn(k)
    return super(DomainHandlerDict, self).__getitem__(k)

  def get(self, k, d=None):
    self._warn(k)
    return super(DomainHandlerDict, self).get(k, d)


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
    domain_handler_dict = handlers.setdefault(
        domain,
        DomainHandlerDict(
            domain or "ai.onnx",
            unknown_message="Unknown op {1} in domain `{0}`. "
            "Can't check specification by ONNX. "
            "Please set should_check flag to False "
            "when call make_node method in handler."))

    since_version = 1
    if handler.ONNX_OP and defs.has(handler.ONNX_OP, domain=handler.DOMAIN):
      since_version = defs.get_schema(
          handler.ONNX_OP, domain=handler.DOMAIN,
          max_inclusive_version=version).since_version
    else:
      for tf_op in handler.TF_OP:
        domain_handler_dict.unknown[tf_op] = handler.ONNX_OP or tf_op
    handler.SINCE_VERSION = since_version

    for tf_op in handler.TF_OP:
      domain_handler_dict[tf_op] = handler
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
    domain_handler_dict = handlers.setdefault(
        domain,
        DomainHandlerDict(
            domain or "ai.onnx",
            failed_message="Fail to get since_version of {1} in domain `{0}` "
            "with max_inclusive_version={2}. Set to 1.",
            unknown_message="Unknown op {1} in domain `{0}`."))

    since_version = 1
    if defs.has(handler.ONNX_OP, domain=handler.DOMAIN):
      try:
        since_version = defs.get_schema(
            handler.ONNX_OP,
            domain=handler.DOMAIN,
            max_inclusive_version=version).since_version
      except RuntimeError:
        domain_handler_dict.failed[handler.ONNX_OP] = version
    else:
      domain_handler_dict.unknown[handler.ONNX_OP] = handler.ONNX_OP
    handler.SINCE_VERSION = since_version
    domain_handler_dict[handler.ONNX_OP] = handler
  return handlers


def get_frontend_coverage():
  """ Get frontend coverage for document.

  :return: onnx_coverage: e.g. {'domain': {'ONNX_OP': [versions], ...}, ...}
  tf_coverage: e.g. {'domain': {'TF_OP': [versions], ...}, ...}
  """

  tf_coverage = {}
  onnx_coverage = {}
  experimental_op = set()
  for handler in FrontendHandler.__subclasses__():
    handler.check_cls()
    versions = handler.get_versions()
    domain = handler.DOMAIN
    for tf_op in handler.TF_OP:
      _update_coverage(tf_coverage, domain, tf_op, versions)
    if handler.ONNX_OP:
      onnx_op = handler.ONNX_OP
      if getattr(handler, "EXPERIMENTAL", False):
        experimental_op.add(handler.ONNX_OP)
      _update_coverage(onnx_coverage, domain, onnx_op, versions)
  return onnx_coverage, tf_coverage, experimental_op


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
