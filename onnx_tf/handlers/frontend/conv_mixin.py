from onnx.helper import make_node

from onnx_tf.common import get_unique_suffix
from onnx_tf.handlers.frontend.transpose import Transpose


class ConvMixin(object):

  @classmethod
  def conv_op(cls, node, d=2, is_depthwise=False, **kwargs):
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
    n_groups = 1
    if is_depthwise:
      n_groups = kernel_shape[-1]
    output_shape = list(
        map(lambda i: node.attr["_output_shapes"][0][i], spatial_indices))
    input_shape = list(
        map(lambda i: node_dict[node.inputs[0]].attr["_output_shapes"][0][i],
            spatial_indices))
    pads = cls.cal_pads(auto_pad, len(spatial_indices), input_shape,
                        output_shape, strides, kernel_shape)

    w_unique_suffix = get_unique_suffix()
    w_transpose_node = Transpose.handle_node_proto(
        make_node(
            "Transpose", [node.inputs[1], "perm"],
            [node.inputs[1] + "_T_" + w_unique_suffix],
            name=node.inputs[1] + "_T_" + w_unique_suffix),
        consts={"perm": [d + 1, d] + list(range(d))})

    conv_node = cls.make_node_from_tf_node(
        node, [node.inputs[0], w_transpose_node.output[0]],
        pads=pads,
        group=n_groups,
        kernel_shape=kernel_shape,
        strides=strides,
        dilations=dilations,
        data_format_auto_convert=True)

    if not isinstance(conv_node, list):
      conv_node = [conv_node]
    return [w_transpose_node] + conv_node

  @staticmethod
  def cal_pads(auto_pad, spatial_dim, input_shape, output_shape, strides,
               kernel_shape):
    pads = [0] * spatial_dim * 2
    if auto_pad == "SAME_UPPER":
      for i in range(spatial_dim):
        pad_shape = (
            output_shape[i] - 1) * strides[i] + kernel_shape[i] - input_shape[i]
        pads[i] = pad_shape // 2
        pads[i + spatial_dim] = pad_shape - pad_shape // 2
    return pads
