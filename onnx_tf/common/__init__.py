from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import re
import sys
import uuid
import warnings

from onnx.backend.base import DeviceType
from tensorflow.python.client import device_lib

IS_PYTHON3 = sys.version_info > (3,)


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

  if supports_device("CUDA"):
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


def get_output_node_names(graph_def):
  """Get output node names from GraphDef.

  Args:
    graph_def: GraphDef object.

  Returns:
    List of output node names.
  """
  nodes, input_names = dict(), set()
  for node in graph_def.node:
    nodes[node.name] = node
    input_names.update(set(node.input))
  return list(set(nodes) - input_names)


class Deprecated:
  """Add deprecated message when function is called.

  Usage:
    from onnx_tf.common import deprecated

    @deprecated
    def func():
      pass

    UserWarning: func is deprecated. It will be removed in future release.

    @deprecated({"arg": "Message", "arg_1": deprecated.REMOVE})
    def func(arg, arg_1, arg_2):
      pass

    UserWarning: arg of func is deprecated. Message
    UserWarning: arg_1 of func is deprecated. It will be removed in future release.
  """

  REMOVE = " It will be removed in future release."

  def __call__(self, *args, **kwargs):
    return self.deprecated_decorator(*args, **kwargs)

  @staticmethod
  def deprecated_decorator(arg=None):
    arg = arg or {}

    if inspect.isfunction(arg):
      warnings.warn("{} is deprecated.{}".format(
          arg.__module__ + "." + arg.__name__, Deprecated.REMOVE))
      return arg
    else:
      name_message_dict = arg

      def deco(func):
        for name, message in name_message_dict.items():
          warnings.warn("{} of {} is deprecated.{}".format(
              name, func.__module__ + "." + func.__name__, message))
        return func

      return deco


deprecated = Deprecated()
