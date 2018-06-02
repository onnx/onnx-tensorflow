from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import sys
import uuid

from onnx.backend.base import DeviceType
from tensorflow.python.client import device_lib
from tensorflow.python.framework.tensor_util import MakeNdarray

from onnx_tf.common import data_type


# This function inserts an underscore before every upper
# case letter and lowers that upper case letter except for
# the first letter.
def op_name_to_lower(name):
  return re.sub('(?<!^)(?=[A-Z])', '_', name).lower()


class AttrConverter(object):

  @classmethod
  def tf2onnx(cls, attr):
    return cls._convert_tf_attr_value(attr)

  @classmethod
  def onnx2tf(cls, attr):
    return cls._convert_onnx_attribute_proto(attr)

  @staticmethod
  def _convert_tf_attr_value(attr):
    """ convert Tensorflow AttrValue object to Python object
    """
    if attr.HasField('list'):
      return AttrConverter._convert_tf_list_value(attr.list)
    if attr.HasField('s'):
      return attr.s
    elif attr.HasField('i'):
      return attr.i
    elif attr.HasField('f'):
      return attr.f
    elif attr.HasField('b'):
      return attr.b
    elif attr.HasField('type'):
      return attr.type
    elif attr.HasField('shape'):
      return attr.type
    elif attr.HasField('tensor'):
      return attr.tensor
    else:
      raise ValueError("Unsupported Tensorflow attribute: {}".format(attr))

  @staticmethod
  def _convert_tf_list_value(list_value):
    """ convert Tensorflow ListValue object to Python object
    """
    if list_value.s:
      return list_value.s
    elif list_value.i:
      return list_value.i
    elif list_value.f:
      return list_value.f
    elif list_value.b:
      return list_value.b
    elif list_value.tensor:
      return list_value.tensor
    elif list_value.type:
      return list_value.type
    elif list_value.shape:
      return list_value.shape
    elif list_value.func:
      return list_value.func
    else:
      raise ValueError(
          "Unsupported Tensorflow attribute: {}".format(list_value))

  @staticmethod
  def _convert_onnx_attribute_proto(attr_proto):
    """
    Convert an ONNX AttributeProto into an appropriate Python object
    for the type.
    NB: Tensor attribute gets returned as the straight proto.
    """
    if attr_proto.HasField('f'):
      return attr_proto.f
    elif attr_proto.HasField('i'):
      return attr_proto.i
    elif attr_proto.HasField('s'):
      return str(attr_proto.s,
                 'utf-8') if sys.version_info[0] >= 3 else attr_proto.s
    elif attr_proto.HasField('t'):
      return attr_proto.t  # this is a proto!
    elif attr_proto.floats:
      return list(attr_proto.floats)
    elif attr_proto.ints:
      return list(attr_proto.ints)
    elif attr_proto.strings:
      str_list = list(attr_proto.strings)
      if sys.version_info[0] >= 3:
        str_list = map(lambda x: str(x, 'utf-8'), str_list)
      return str_list
    else:
      raise ValueError("Unsupported ONNX attribute: {}".format(attr_proto))


class AttrTranslator(object):

  # Keyed by old attribute names.
  _tf_attr_translator = {
    "_output_shapes": lambda x: list(map(lambda shape: AttrTranslator.get_tf_shape_as_list(shape.dim), x.list.shape)),
    "shape": lambda x: AttrTranslator.get_tf_shape_as_list(x.shape.dim),
    "T": lambda x: data_type.tf2onnx(x.type),
    "dtype": lambda x: data_type.tf2onnx(x.type),
    "value": lambda x: MakeNdarray(x.tensor),
    "seed2": lambda x: float(x.i),
    "seed": lambda x: float(x.i),
    "keep_dims": lambda x: int(x.b),
    "squeeze_dims": lambda x: list(x.list.i),
  }

  _onnx_attr_translator = {
      "axis": lambda x: int(x),
      "axes": lambda x: [int(a) for a in x],
      "dtype": lambda x: data_type.onnx2tf(x),
      "keepdims": lambda x: bool(x),
      "to": lambda x: data_type.onnx2tf(x),
  }

  @classmethod
  def translate_tf(cls, key, val):
    return cls._tf_attr_translator.get(key, lambda x: x)(val)

  @classmethod
  def translate_onnx(cls, key, val):
    return cls._onnx_attr_translator.get(key, lambda x: x)(val)

  @staticmethod
  def get_tf_shape_as_list(tf_shape_dim):
    return list(map(lambda x: x.size, list(tf_shape_dim)))


def get_unique_suffix():
  return str(uuid.uuid4())[:8]


def get_perm_from_formats(_from, _to):
  return list(map(lambda x: _from.find(x), _to))


# TODO: allow more flexible placement
def get_device_option(device):
  m = {DeviceType.CPU: '/cpu', DeviceType.CUDA: '/gpu'}
  return m[device.type]


def get_data_format(x_rank, support_cuda):
  sp_dim_names = ["D", "H", "W"]
  sp_dim_lst = []
  for i in range(x_rank - 2):
    sp_dim_lst.append(sp_dim_names[-i - 1])

  sp_dim_string = "".join(reversed(sp_dim_lst))
  storage_format = "NC" + sp_dim_string

  if support_cuda:
    compute_format = "NC" + sp_dim_string
  else:
    compute_format = "N" + sp_dim_string + "C"
  return storage_format, compute_format


def supports_device(device):
  if device == "CUDA":
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'
               ]) > 0
  elif device == "CPU":
    return True
  return False


# Constant string used to indicate that requested padding
# is not natively supported in Tensorflow.
PAD_TF_INCOMPATIBLE = "PAD_TF_INCOMPATIBLE"

attr_converter = AttrConverter()
attr_translator = AttrTranslator()
