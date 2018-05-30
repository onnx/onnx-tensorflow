import warnings

from onnx import defs

from onnx_tf.handlers.frontend import *  # noqa
from onnx_tf.handlers.frontend_handler import FrontendHandler
from . import op_name_to_lower


def get_all_frontend_handlers(opset_dict):
  """ Get a dict of all frontend handler classes.
  e.g. {'domain': {'Abs': Abs handler class}, ...}, }.

  :param opset_dict: A dict of opset. e.g. {'domain': version, ...}
  :return: Dict.
  """
  handlers = {}
  for handler in FrontendHandler.__subclasses__():
    handler.check()

    domain = handler.DOMAIN
    version = opset_dict[domain]
    handler.VERSION = version

    since_version = 1
    if defs.has(handler.ONNX_OP, domain=handler.DOMAIN):
      since_version = defs.get_schema(
          handler.ONNX_OP, domain=handler.DOMAIN,
          max_inclusive_version=version).since_version
    else:
      warnings.warn("Unknown op {} in domain `{}`. "
                    "Can't check specification by ONNX. "
                    "Please set should_check flag to False "
                    "when call make_node method in handler.".format(
                        handler.ONNX_OP, handler.DOMAIN or "ai.onnx"))
    handler.SINCE_VERSION = since_version

    for tf_op in handler.TF_OP:
      handlers.setdefault(domain, {})[tf_op] = handler
  return handlers


def get_frontend_coverage():
  """ Get frontend coverage. For document.

  :return: onnx_coverage: e.g. {'domain': {'ONNX_OP': [versions], ...}, ...}
  tf_coverage: e.g. {'domain': {'TF_OP': [versions], ...}, ...}
  """

  def _update_coverage(coverage, domain, key, versions):
    domain_coverage = coverage.setdefault(domain, {})
    vers = domain_coverage.get(key, [])
    vers.extend(versions)
    domain_coverage[key] = sorted(list(set(vers)))

  tf_coverage = {}
  onnx_coverage = {}
  for handler in FrontendHandler.__subclasses__():
    handler.check()

    versions = handler.get_versions()
    domain = handler.DOMAIN
    for tf_op in handler.TF_OP:
      _update_coverage(tf_coverage, domain, op_name_to_lower(tf_op), versions)
    _update_coverage(onnx_coverage, domain, handler.ONNX_OP, versions)
  return onnx_coverage, tf_coverage
