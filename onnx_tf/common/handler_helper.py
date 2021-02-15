from onnx import defs

from onnx_tf.handlers.backend import *  # noqa
from onnx_tf.handlers.backend_handler import BackendHandler
import onnx_tf.common as common

def common_logging(handler, version):
  common.logger.debug("Fail to get since_version of {} in domain `{}` "
                      "with max_inclusive_version={}. Set to 1.".format(
                          handler.ONNX_OP, handler.DOMAIN, version))

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
    version = opset_dict[domain] if domain in opset_dict else 1
    handler.VERSION = version

    since_version = 1
    if defs.has(handler.ONNX_OP, domain=handler.DOMAIN):
      try:
        since_version = defs.get_schema(
            handler.ONNX_OP,
            domain=handler.DOMAIN,
            max_inclusive_version=version).since_version
      except RuntimeError:
        # ONNX throws RuntimeError up to 1.8
        common_logging(handler, version)
      except defs.SchemaError:
        # ONNX changed to defs.SchemaError since 1.9
        common_logging(handler, version)
    else:
      common.logger.debug("Unknown op {} in domain `{}`.".format(
          handler.ONNX_OP, handler.DOMAIN or "ai.onnx"))
    handler.SINCE_VERSION = since_version
    handlers.setdefault(domain, {})[handler.ONNX_OP] = handler
  return handlers


def get_backend_coverage():
  """ Get backend coverage for document.

  :return: onnx_coverage: e.g. {'domain': {'ONNX_OP': [versions], ...}, ...}
  """

  onnx_coverage = {}
  experimental_op = set()
  for handler in BackendHandler.__subclasses__():
    handler.check_cls()

    versions = handler.get_versions()
    domain = handler.DOMAIN
    if getattr(handler, "EXPERIMENTAL", False):
      experimental_op.add(handler.ONNX_OP)
    _update_coverage(onnx_coverage, domain, handler.ONNX_OP, versions)
  return onnx_coverage, experimental_op


def _update_coverage(coverage, domain, key, versions):
  domain_coverage = coverage.setdefault(domain, {})
  vers = domain_coverage.get(key, [])
  vers.extend(versions)
  domain_coverage[key] = sorted(list(set(vers)))


def get_backend_partial_support_detail():
  ps_dict = {}
  opset_dict = dict([(defs.ONNX_DOMAIN, defs.onnx_opset_version())])
  handlers = get_all_backend_handlers(opset_dict)[defs.ONNX_DOMAIN]
  for op_name in handlers:
    if handlers[op_name].PARTIAL_SUPPORT:
      ps_dict[op_name] = handlers[op_name].PS_DESCRIPTION
  return ps_dict
