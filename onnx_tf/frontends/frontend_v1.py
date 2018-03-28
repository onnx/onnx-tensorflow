"""Frontend for exporting Tensorflow graph to ONNX graph

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from onnx_tf.frontend import TensorflowFrontendBase
from onnx import helper, mapping


class TensorflowFrontend(TensorflowFrontendBase):
  """ Tensorflow Frontend for ONNX
  """

  ONNX_TO_HANDLER = {
      "add": "bias_add",
      "and": "logical_and",
      "batch_normalization": "fused_batch_norm",
      "conv": ["conv1_d", "conv2_d", "conv3_d"],
      "average_pool": "avg_pool",
      "max_pool": "max_pool",
      "or": "logical_or",
      "pad": "pad",
      "random_normal": "random_standard_normal",
      "random_uniform": "random_uniform",
      "reduce_max": "max",
      "reduce_mean": "mean",
      "reduce_min": "min",
      "reduce_prod": "prod",
      "reduce_sum": "sum",
      "reshape": "reshape",
      "split": "split_v",
      "squeeze": "squeeze",
      "sub": "sub",
      "transpose": "transpose",
      "xor": "logical_xor",
      "concat": "concat_v2",
  }

  @classmethod
  def handle_avg_pool(cls, node, **kwargs):
    return cls._pool_op(node, "AveragePool", **kwargs)

  @classmethod
  def handle_fused_batch_norm(cls, node, **kwargs):
    return helper.make_node(
        "BatchNormalization",
        node.inputs, [node.name],
        epsilon=node.attr.get("epsilon", 1e-5),
        is_test=node.attr.get("is_training", 0),
        consumed_inputs=node.attr.get("consumed_inputs", [0, 0, 0, 1, 1]))

  @classmethod
  def handle_bias_add(cls, node, **kwargs):
    return cls._bin_op(node, "Add")

  @classmethod
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
    consts_proto = kwargs["consts_proto"]
    output_shapes = kwargs["output_shapes"]
    kernel_name = node.inputs[1].replace("/read", "")
    kernel_shape = consts_proto[kernel_name].dims[:d]
    dims = list(range(len(consts_proto[kernel_name].dims)))
    output_shape = list(
        map(lambda i: node.attr["_output_shapes"][0][i], spatial_indices))
    input_shape = list(
        map(lambda i: output_shapes[node.inputs[0]][0][i], spatial_indices))
    pads = cls._cal_pads(auto_pad, len(spatial_indices), input_shape,
                         output_shape, strides, kernel_shape)
    # Copy weight
    new_kernel_name = node.inputs[1].replace("/read", "/transposed/read")
    kernel = consts_proto[kernel_name]
    field = mapping.STORAGE_TENSOR_TYPE_TO_FIELD[
        mapping.TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE[kernel.data_type]]
    transposed_vals = np.transpose(
        np.reshape(np.array(getattr(kernel, field)), kernel.dims),
        axes=dims[-2:][::-1] + dims[:len(dims) - 2])
    transposed_kernel = helper.make_tensor(
      name=new_kernel_name.replace("/read", ""),
      data_type=kernel.data_type,
      dims=transposed_vals.shape,
      vals=transposed_vals.flatten().tolist()
    )
    kwargs["additional_consts_proto"].append(transposed_kernel)

    kernel_node = helper.make_node(
        "Identity", [new_kernel_name.replace("/read", "")], [new_kernel_name])
    return [
        kernel_node,
        helper.make_node(
            "Conv", [node.inputs[0], kernel_node.output[0]], [node.name],
            pads=pads,
            kernel_shape=kernel_shape,
            strides=strides,
            dilations=dilations)
    ]

  @classmethod
  def handle_conv1_d(cls, node, **kwargs):
    return cls._conv(node, 1, **kwargs)

  @classmethod
  def handle_conv2_d(cls, node, **kwargs):
    return cls._conv(node, 2, **kwargs)

  @classmethod
  def handle_conv3_d(cls, node, **kwargs):
    return cls._conv(node, 3, **kwargs)

  @classmethod
  def handle_logical_and(cls, node, **kwargs):
    return cls._bin_op(node, "And")

  @classmethod
  def handle_logical_or(cls, node, **kwargs):
    return cls._bin_op(node, "Or")

  @classmethod
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
  def handle_max(cls, node, **kwargs):
    return cls._reduce_op("ReduceMax", node, **kwargs)

  @classmethod
  def handle_max_pool(cls, node, **kwargs):
    return cls._pool_op(node, "MaxPool", **kwargs)

  @classmethod
  def handle_mean(cls, node, **kwargs):
    return cls._reduce_op("ReduceMean", node, **kwargs)

  @classmethod
  def handle_min(cls, node, **kwargs):
    return cls._reduce_op("ReduceMin", node, **kwargs)

  @classmethod
  def handle_prod(cls, node, **kwargs):
    return cls._reduce_op("ReduceProd", node, **kwargs)

  @classmethod
  def handle_sum(cls, node, **kwargs):
    return cls._reduce_op("ReduceSum", node, **kwargs)

  @classmethod
  def handle_reshape(cls, node, **kwargs):
    consts = kwargs["consts"]
    assert node.inputs[1] in consts.keys()
    shape = consts[node.inputs[1]]
    return helper.make_node(
        "Reshape", [node.inputs[0]], [node.name], shape=shape)

  @classmethod
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
  def handle_squeeze(cls, node, **kwargs):
    assert "squeeze_dims" in node.attr.keys(), ("Squeeze dims have to be"
                                                "specified")
    axes = node.attr["squeeze_dims"]
    return helper.make_node("Squeeze", [node.inputs[0]], [node.name], axes=axes)

  @classmethod
  def handle_sub(cls, node, **kwargs):
    return cls._bin_op(node, "Sub")

  @classmethod
  def handle_transpose(cls, node, **kwargs):
    consts = kwargs["consts"]
    perm = consts[node.inputs[1]]
    return helper.make_node(
        "Transpose", [node.inputs[0]], [node.name], perm=perm)

  @classmethod
  def handle_logical_xor(cls, node, **kwargs):
    return cls._bin_op(node, "Xor")
