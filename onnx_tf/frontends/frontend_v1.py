"""Frontend for exporting Tensorflow graph to ONNX graph

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from onnx import helper
from onnx import TensorProto
from onnx_tf.common import as_dtype
from onnx_tf.common import get_unique_suffix
from onnx_tf.common import get_perm_from_formats
from onnx_tf.common import TF_TYPE_TO_ONNX_TYPE
from onnx_tf.frontend import TensorflowFrontendBase

register_onnx_op = TensorflowFrontendBase.register_onnx_op


class TensorflowFrontend(TensorflowFrontendBase):
  """ Tensorflow Frontend for ONNX
  """

  @classmethod
  @register_onnx_op("ArgMax")
  def handle_arg_max(cls, node, **kwargs):
    axis = np.asscalar(kwargs["consts"][node.inputs[1]])
    return helper.make_node(
        "ArgMax", [node.inputs[0]], [node.name], axis=axis, keepdims=0)

  @classmethod
  @register_onnx_op("ArgMin")
  def handle_arg_min(cls, node, **kwargs):
    axis = np.asscalar(kwargs["consts"][node.inputs[1]])
    return helper.make_node(
        "ArgMin", [node.inputs[0]], [node.name], axis=axis, keepdims=0)

  @classmethod
  @register_onnx_op("AveragePool")
  def handle_avg_pool(cls, node, **kwargs):
    return cls._pool_op(node, "AveragePool", **kwargs)

  @classmethod
  @register_onnx_op("BatchNormalization")
  def handle_fused_batch_norm(cls, node, **kwargs):
    return helper.make_node(
        "BatchNormalization",
        node.inputs, [node.name],
        epsilon=node.attr.get("epsilon", 1e-5),
        is_test=node.attr.get("is_training", 0),
        consumed_inputs=node.attr.get("consumed_inputs", [0, 0, 0, 1, 1]))

  @classmethod
  @register_onnx_op("Add")
  def handle_bias_add(cls, node, **kwargs):
    data_format = node.attr.get("data_format", "NHWC")
    channel_first = data_format[1] == "C"
    axis = 1 if channel_first else -1
    return cls._bin_op(node, "Add", axis=axis)

  @classmethod
  @register_onnx_op("Cast")
  def handle_cast(cls, node, **kwargs):
    dst_t = TF_TYPE_TO_ONNX_TYPE[as_dtype(node.attr["DstT"])]
    return helper.make_node("Cast", [node.inputs[0]], [node.name], to=dst_t)

  @classmethod
  @register_onnx_op("Concat")
  def handle_concat_v2(cls, node, **kwargs):
    consts = kwargs["consts"]
    assert node.inputs[-1] in consts.keys()
    axis = int(consts[node.inputs[-1]])
    return helper.make_node(
        "Concat", inputs=node.inputs[0:-1], outputs=[node.name], axis=axis)

  @classmethod
  def _conv(cls, node, d, **kwargs):
    auto_pad = node.attr["padding"].decode("UTF-8")
    auto_pad = "SAME_UPPER" if auto_pad == "SAME" else auto_pad
    data_format = node.attr["data_format"].decode("UTF-8")
    spatial_indices = [
        i for i in range(len(data_format)) if data_format[i] not in ["N", "C"]
    ]
    strides = list(map(lambda i: node.attr["strides"][i], spatial_indices))
    dilations = list(
        map(lambda i: node.attr.get("dilations", [1] * (d + 2))[i],
            spatial_indices))
    node_dict = kwargs["node_dict"]
    kernel_shape = node_dict[node.inputs[1]].attr["_output_shapes"][0][:d]
    output_shape = list(
        map(lambda i: node.attr["_output_shapes"][0][i], spatial_indices))
    input_shape = list(
        map(lambda i: node_dict[node.inputs[0]].attr["_output_shapes"][0][i],
            spatial_indices))
    pads = cls._cal_pads(auto_pad, len(spatial_indices), input_shape,
                         output_shape, strides, kernel_shape)
    unique_suffix = get_unique_suffix()
    transpose_node = helper.make_node(
        "Transpose", [node.inputs[1]], [node.inputs[1] + "_T_" + unique_suffix],
        perm=[d + 1, d] + list(range(d)))
    conv_node = helper.make_node(
        "Conv", [node.inputs[0], node.inputs[1] + "_T_" + unique_suffix],
        [node.name],
        pads=pads,
        kernel_shape=kernel_shape,
        strides=strides,
        dilations=dilations)

    return [transpose_node, conv_node]

  @classmethod
  @register_onnx_op("Conv")
  def handle_conv1_d(cls, node, **kwargs):
    return cls._conv(node, 1, **kwargs)

  @classmethod
  @register_onnx_op("Conv")
  def handle_conv2_d(cls, node, **kwargs):
    return cls._conv(node, 2, **kwargs)

  @classmethod
  @register_onnx_op("Conv")
  def handle_conv3_d(cls, node, **kwargs):
    return cls._conv(node, 3, **kwargs)

  @classmethod
  @register_onnx_op("ConstantFill")
  def handle_fill(cls, node, **kwargs):
    value = float(np.asscalar(kwargs["consts"][node.inputs[1]]))
    return helper.make_node(
        "ConstantFill", [node.inputs[0]], [node.name],
        input_as_shape=1,
        value=value)

  @classmethod
  @register_onnx_op("Pad")
  def handle_pad(cls, node, **kwargs):
    consts = kwargs["consts"]
    assert node.inputs[1] in consts.keys()
    supported_modes = ["constant", "reflect"]
    mode = node.attr.get("mode", "constant")
    assert mode.lower() in supported_modes
    pads = np.transpose(consts[node.inputs[1]]).flatten()

    return helper.make_node(
        "Pad", [node.inputs[0]], [node.name],
        name=node.name,
        paddings=pads,
        mode=mode,
        value=0.0)

  @classmethod
  @register_onnx_op("RandomNormal")
  def handle_random_standard_normal(cls, node, **kwargs):
    """ Tensorflow does not have a generic random_normal op.
        The generic random_normal op is translated into a scaled
        and offsetted random standard normal op.
    """
    return helper.make_node(
        "RandomNormal", [], [node.name],
        dtype=node.attr["dtype"],
        seed=node.attr["seed"],
        mean=0.0,
        scale=1.0,
        shape=node.attr["_output_shapes"][0])

  @classmethod
  @register_onnx_op("RandomUniform")
  def handle_random_uniform(cls, node, **kwargs):
    """ Tensorflow does not have a generic random_uniform op.
        The generic random_uniform op is translated into a scaled
        and offsetted random standard uniform op.
    """
    return helper.make_node(
        "RandomUniform", [], [node.name],
        dtype=node.attr["dtype"],
        seed=node.attr["seed"],
        high=1.0,
        low=0.0,
        shape=node.attr["_output_shapes"][0])

  @classmethod
  @register_onnx_op("ReduceMax")
  def handle_max(cls, node, **kwargs):
    return cls._reduce_op("ReduceMax", node, **kwargs)

  @classmethod
  @register_onnx_op("Max")
  def handle_maximum(cls, node, **kwargs):
    return helper.make_node("Max", node.inputs, [node.name], name=node.name)

  @classmethod
  @register_onnx_op("MaxPool")
  def handle_max_pool(cls, node, **kwargs):
    return cls._pool_op(node, "MaxPool", **kwargs)

  @classmethod
  @register_onnx_op("ReduceMean")
  def handle_mean(cls, node, **kwargs):
    return cls._reduce_op("ReduceMean", node, **kwargs)

  @classmethod
  @register_onnx_op("ReduceMin")
  def handle_min(cls, node, **kwargs):
    return cls._reduce_op("ReduceMin", node, **kwargs)

  @classmethod
  @register_onnx_op("ReduceProd")
  def handle_prod(cls, node, **kwargs):
    return cls._reduce_op("ReduceProd", node, **kwargs)

  @classmethod
  @register_onnx_op("ReduceSum")
  def handle_sum(cls, node, **kwargs):
    return cls._reduce_op("ReduceSum", node, **kwargs)

  @classmethod
  @register_onnx_op("Reshape")
  def handle_reshape(cls, node, **kwargs):
    consts = kwargs["consts"]
    assert node.inputs[1] in consts.keys()
    shape = consts[node.inputs[1]]
    return helper.make_node(
        "Reshape", [node.inputs[0]], [node.name], shape=shape)

  @classmethod
  @register_onnx_op("SpaceToDepth")
  def handle_space_to_depth(cls, node, **kwargs):
    blocksize = node.attr["block_size"]
    data_format = node.attr.get("data_format", "NHWC").decode()

    assert data_format in ["NHWC", "NCHW"], \
      ("data format {} should be in ['NCHW', 'NHWC'].".format(data_format))

    if data_format == "NHWC":
      transpose_unique_suffix = get_unique_suffix()
      space_to_depth_unique_suffix = get_unique_suffix()
      transpose_name = node.inputs[0] + "_T_" + transpose_unique_suffix
      space_to_depth_name = node.inputs[0] + "_T_STD_" + space_to_depth_unique_suffix
      before_transpose_node = helper.make_node(
          "Transpose",
          [node.inputs[0]], [transpose_name],
          perm=get_perm_from_formats("NHWC", "NCHW"))
      space_to_depth_node = helper.make_node(
          "SpaceToDepth",
          [transpose_name], [space_to_depth_name], blocksize=blocksize)

      after_transpose_node = helper.make_node(
          "Transpose",
          [space_to_depth_name], [node.name],
          perm=get_perm_from_formats("NCHW", "NHWC"))

      return [before_transpose_node, space_to_depth_node, after_transpose_node]

    if data_format == "NCHW":
      return helper.make_node(
          "SpaceToDepth", [node.inputs[0]], [node.name], blocksize=blocksize)

  @classmethod
  @register_onnx_op("Split")
  def handle_split_v(cls, node, **kwargs):
    consts = kwargs["consts"]
    split = consts[node.inputs[1]]
    axis = int(consts[node.inputs[2]])
    output_names = [
        node.name + ":{}".format(i) if i > 0 else node.name
        for i in range(len(split))
    ]
    return helper.make_node(
        "Split", [node.inputs[0]], output_names, split=split, axis=axis)

  @classmethod
  @register_onnx_op("Squeeze")
  def handle_squeeze(cls, node, **kwargs):
    assert "squeeze_dims" in node.attr.keys(), ("Squeeze dims have to be"
                                                "specified")
    axes = node.attr["squeeze_dims"]
    return helper.make_node("Squeeze", [node.inputs[0]], [node.name], axes=axes)

  @classmethod
  @register_onnx_op("Tile")
  def handle_tile(cls, node, **kwargs):
    data_type_cast_map = kwargs["data_type_cast_map"]
    data_type_cast_map[node.inputs[1]] = TensorProto.INT64
    return helper.make_node("Tile", node.inputs, [node.name])

  @classmethod
  @register_onnx_op("Transpose")
  def handle_transpose(cls, node, **kwargs):
    consts = kwargs["consts"]
    if node.inputs[1] in consts:
      perm = consts[node.inputs[1]]
    else:
      input_rank = len(
          kwargs['node_dict'][node.inputs[0]].attr['_output_shapes'][0])
      perm = list(reversed(range(input_rank)))

    return helper.make_node(
        "Transpose", [node.inputs[0]], [node.name], perm=perm)
