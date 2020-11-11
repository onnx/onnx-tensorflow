from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import re
import sys
import uuid
import warnings
import logging

from onnx.backend.base import DeviceType
from tensorflow.python.client import device_lib

IS_PYTHON3 = sys.version_info > (3,)
logger = logging.getLogger('onnx-tf')

# create console handler and formatter for logger
console = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)


class SysConfig:

  def __init__(self):
    self.auto_cast = False
    self.device = 'CPU'


sys_config = SysConfig()


class Deprecated:
  """Add deprecated message when function is called.

  Usage:
    from onnx_tf.common import deprecated

    @deprecated
    def func():
      pass

    UserWarning: func is deprecated. It will be removed in future release.

    @deprecated("Message")
    def func():
      pass

    UserWarning: Message

    @deprecated({"arg": "Message",
                 "arg_1": deprecated.MSG_WILL_REMOVE,
                 "arg_2": "",})
    def func(arg, arg_1, arg_2):
      pass

    UserWarning: Message
    UserWarning: arg_1 of func is deprecated. It will be removed in future release.
    UserWarning: arg_2 of func is deprecated.
  """

  MSG_WILL_REMOVE = " It will be removed in future release."

  def __call__(self, *args, **kwargs):
    return self.deprecated_decorator(*args, **kwargs)

  @staticmethod
  def messages():
    return {v for k, v in inspect.getmembers(Deprecated) if k.startswith("MSG")}

  @staticmethod
  def deprecated_decorator(arg=None):
    # deprecate function with default message MSG_WILL_REMOVE
    # @deprecated
    if inspect.isfunction(arg):

      def wrapper(*args, **kwargs):
        warnings.warn("{} is deprecated.{}".format(
            arg.__module__ + "." + arg.__name__, Deprecated.MSG_WILL_REMOVE))
        return arg(*args, **kwargs)

      return wrapper

    deprecated_arg = arg if arg is not None else Deprecated.MSG_WILL_REMOVE

    def deco(func):
      # deprecate arg
      # @deprecated({...})
      if isinstance(deprecated_arg, dict):
        for name, message in deprecated_arg.items():
          if message in Deprecated.messages():
            message = "{} of {} is deprecated.{}".format(
                name, func.__module__ + "." + func.__name__, message or "")
          warnings.warn(message)
      # deprecate function with message
      # @deprecated("message")
      elif isinstance(deprecated_arg, str):
        message = deprecated_arg
        if message in Deprecated.messages():
          message = "{} is deprecated.{}".format(
              func.__module__ + "." + func.__name__, message)
        warnings.warn(message)
      return func

    return deco


deprecated = Deprecated()


# This function inserts an underscore before every upper
# case letter and lowers that upper case letter except for
# the first letter.
def op_name_to_lower(name):
  return re.sub('(?<!^)(?=[A-Z])', '_', name).lower()


def get_unique_suffix():
  """ Get unique suffix by using first 8 chars from uuid.uuid4
  to make unique identity name.

  :return: Unique suffix string.
  """
  return str(uuid.uuid4())[:8]


def get_perm_from_formats(from_, to_):
  """ Get perm from data formats.
  For example:
    get_perm_from_formats('NHWC', 'NCHW') = [0, 3, 1, 2]

  :param from_: From data format string.
  :param to_: To data format string.
  :return: Perm. Int list.
  """
  return list(map(lambda x: from_.find(x), to_))


# TODO: allow more flexible placement
def get_device_option(device):
  m = {DeviceType.CPU: '/cpu', DeviceType.CUDA: '/gpu'}
  return m[device.type]


def get_data_format(x_rank):
  """ Get data format by input rank.
  Channel first if support CUDA.

  :param x_rank: Input rank.
  :return: Data format.
  """
  sp_dim_names = ["D", "H", "W"]
  sp_dim_lst = []
  for i in range(x_rank - 2):
    sp_dim_lst.append(sp_dim_names[-i - 1])

  sp_dim_string = "".join(reversed(sp_dim_lst))
  storage_format = "NC" + sp_dim_string

  if sys_config.device == "CUDA":
    compute_format = "NC" + sp_dim_string
  else:
    compute_format = "N" + sp_dim_string + "C"
  return storage_format, compute_format


def supports_device(device):
  """ Check if support target device.
  :param device: CUDA or CPU.
  :return: If supports.
  """
  if device == "CUDA":
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'
               ]) > 0
  elif device == "CPU":
    return True
  return False


def get_variable_name(node, var_name):
  """ Get variable name.
  :param node: ONNX NodeProto object
  :param var_name: name of the variable
  :return: unique variable name
  """
  v_name = node.op_type.lower() + '_' + var_name
  return v_name + '_' + node.name.lower() if node.name else v_name


CONST_MINUS_ONE_INT32 = "_onnx_tf_internal_minus_one_int32"
CONST_ZERO_INT32 = "_onnx_tf_internal_zero_int32"
CONST_ONE_INT32 = "_onnx_tf_internal_one_int32"
CONST_ONE_FP32 = "_onnx_tf_internal_one_fp32"
