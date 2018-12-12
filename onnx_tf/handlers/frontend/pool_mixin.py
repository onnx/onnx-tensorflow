from .conv_mixin import ConvMixin


class PoolMixin(object):

  @classmethod
  def args_check(cls, node, **kwargs):
    if "count_include_pad" in kwargs:
      if cls.ONNX_OP != "AveragePool":
        raise RuntimeError("count_include_pad is only for AveragePool.")
      if cls.SINCE_VERSION < 7:
        raise RuntimeError("count_include_pad is added since version 7.")

  @classmethod
  def pool_op(cls, node, **kwargs):
    auto_pad = node.attr["padding"].decode("UTF-8")
    auto_pad = "SAME_UPPER" if auto_pad == "SAME" else auto_pad
    data_format = kwargs.get("data_format",
                             None) or node.attr["data_format"].decode("UTF-8")
    spatial_indices = [
        i for i in range(len(data_format)) if data_format[i] not in ["N", "C"]
    ]
    strides = list(map(lambda i: node.attr["strides"][i], spatial_indices))
    kernel_shape = list(map(lambda i: node.attr["ksize"][i], spatial_indices))
    node_dict = kwargs["node_dict"]
    output_shape = list(
        map(lambda i: node.attr["_output_shapes"][0][i], spatial_indices))
    input_shape = list(
        map(lambda i: node_dict[node.inputs[0]].attr["_output_shapes"][0][i],
            spatial_indices))
    pads = ConvMixin.cal_pads(auto_pad, len(spatial_indices), input_shape,
                              output_shape, strides, kernel_shape)

    node_kwargs = {}
    if "count_include_pad" in kwargs:
      node_kwargs["count_include_pad"] = kwargs["count_include_pad"]
    return cls.make_node_from_tf_node(
        node, [node.inputs[0]],
        pads=pads,
        kernel_shape=kernel_shape,
        strides=strides,
        data_format_auto_convert=True,
        **node_kwargs)
