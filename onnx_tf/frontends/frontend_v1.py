"""Frontend for exporting Tensorflow graph to ONNX graph

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from onnx_tf.frontend import TensorflowFrontendBase
from onnx import helper


class TensorflowFrontend(TensorflowFrontendBase):
  """ Tensorflow Frontend for ONNX
  """

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
            "Pad",
            [node.inputs[0]],
            [node.name],
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
            "RandomNormal",
            [],
            [node.name],
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
            "RandomUniform",
            [],
            [node.name],
            dtype=node.attr["dtype"],
            seed=node.attr["seed"],
            high=1.0,
            low=0.0,
            shape=node.attr["_output_shapes"][0])

  @classmethod
  def handle_max(cls, node, **kwargs):
    return cls._reduce_op("ReduceMax", node, **kwargs)

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
    return helper.make_node("Reshape",
                            [node.inputs[0]],
                            [node.name],
                            shape=shape)

  @classmethod
  def handle_split_v(cls, node, **kwargs):
    consts = kwargs["consts"]
    split = consts[node.inputs[1]]
    axis = int(consts[node.inputs[2]])
    output_names = [node.name + ":{}".format(i) if i>0 else node.name for i in range(len(split))]
    return helper.make_node("Split",
                            [node.inputs[0]],
                            output_names,
                            split=split,
                            axis=axis)

  @classmethod
  def handle_squeeze(cls, node, **kwargs):
    assert "squeeze_dims" in node.attr.keys(), ("Squeeze dims have to be"
      "specified")
    axes = node.attr["squeeze_dims"]
    return helper.make_node("Squeeze",
                            [node.inputs[0]],
                            [node.name],
                            axes=axes)

  @classmethod
  def handle_sub(cls, node, **kwargs):
    return cls._bin_op(node, "Sub")

  @classmethod
  def handle_transpose(cls, node, **kwargs):
    consts = kwargs["consts"]
    perm = consts[node.inputs[1]]
    return helper.make_node("Transpose",
                            [node.inputs[0]],
                            [node.name],
                            perm=perm)

  @classmethod
  def handle_logical_xor(cls, node, **kwargs):
    return cls._bin_op(node, "Xor")

  @classmethod
  def handle_concat_v2(cls, node, **kwargs):
    consts = kwargs["consts"]
    assert node.inputs[-1] in consts.keys()
    axis = int(consts[node.inputs[-1]])
    return helper.make_node("Concat",
                            inputs=node.inputs[0:-1],
                            outputs=[node.name],
                            axis=axis)
