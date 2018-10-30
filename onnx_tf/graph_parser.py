import collections

import numpy as np
import tensorflow as tf

from onnx_tf.common import data_type
from onnx_tf.common import get_unique_suffix
from onnx_tf.pb_wrapper import TensorflowNode


class Parser(object):

  @classmethod
  def parse(cls, nodes):
    return nodes


class MultiRNNParser(Parser):

  class NodeInfoHolder(object):
    """Helper class for holding node info.
    """

    def __init__(self):
      self.scopes = set()
      self.nodes = collections.defaultdict(list)
      self.cell_dict = collections.defaultdict(dict)
      self.nodes_keep = collections.defaultdict(set)

  @classmethod
  def _make_node_info(cls, nodes):
    """Make NodeInfoHolder object.

    Args:
      nodes: List of NodeDef.

    Returns:
      NodeInfoHolder object.

    """
    node_info_holder = cls.NodeInfoHolder()
    for n in nodes:
      scopes = n.name.split("/")
      if "multi_rnn_cell" in scopes:
        idx_multi_rnn_cell = scopes.index("multi_rnn_cell")
        idx_while = scopes.index(
            "while") if "while" in scopes else len(scopes) - 1
        idx_cell_name = idx_multi_rnn_cell + 1
        idx_cell_type = idx_multi_rnn_cell + 2
        scope = "/".join(scopes[:min(idx_multi_rnn_cell, idx_while)])
        node_info_holder.scopes.add(scope)
        cell_no = int(scopes[idx_cell_name].replace("cell_", ""))
        node_info_holder.cell_dict[cell_no]["type"] = scopes[
            idx_cell_type].replace("_cell", "")
        for key in ["kernel", "bias"]:
          if key in scopes[-2:]:
            if key == scopes[-2] and "read" == scopes[-1]:
              node_info_holder.cell_dict[cell_no][key] = n.name
            node_info_holder.nodes_keep[scope].add(n.name)
        prev_c = [i for i in n.input if "cell_{}".format(cell_no - 1) in i]
        if prev_c:
          node_info_holder.cell_dict[cell_no]["prev_c"] = prev_c[0]
    return node_info_holder

  @classmethod
  def _group_nodes(cls, nodes, node_info_holder):
    """Grouping nodes into [nodes, cell_nodes_key, nodes].

    Args:
      nodes: List of NodeDef.
      node_info_holder: NodeInfoHolder object.

    Returns:
      Grouped nodes list.
      Cell nodes dict.

    """
    group_nodes = [[]]
    new_cell_nodes = collections.defaultdict(list)
    for n in nodes:
      scope = [s for s in node_info_holder.scopes if s in n.name]
      if len(scope) != 1:
        raise ValueError(
            "More than one scope {} contained in node name {}.".format(
                str(scope), n.name))
      scope = scope[0]
      if scope:
        if group_nodes[-1]:
          group_nodes.append(scope)
          group_nodes.append([])
        node_info_holder.nodes[scope].append(n)
        if n.name in node_info_holder.nodes_keep[scope]:
          new_cell_nodes[scope].append(n)
      else:
        group_nodes[-1].append(n)
    return group_nodes, new_cell_nodes

  @classmethod
  def parse(cls, nodes):
    """Parse nodes.

    Args:
      nodes: List of NodeDef.

    Returns:
      Parsed nodes of TensorflowNode.

    """
    node_info_holder = cls._make_node_info(nodes)
    node_dict = {n.name: TensorflowNode(n) for n in nodes}
    group_nodes, new_cell_nodes = cls._group_nodes(nodes, node_info_holder)

    for scope in node_info_holder.nodes:
      inputs, outputs = cls._get_input_output_node_names(
          node_info_holder.nodes[scope])
      inputs = [i for i in inputs if scope not in i]
      input_nodes = [node_dict[i] for i in inputs]

      batch_major = [
          n for n in node_info_holder.nodes[scope] if inputs[0] in n.input
      ][0].op == "Transpose"

      if batch_major:
        perm_node, trans_node = cls._make_major_transpose_nodes(
            inputs, scope, node_dict, new_cell_nodes[scope][-1], False)
        input_nodes = [trans_node]
        new_cell_nodes[scope].extend([perm_node, trans_node])

      dtype = input_nodes[0].attr["T"]
      for cell_no, cell_info in node_info_holder.cell_dict.items():
        cell_node = cls._make_rnn_node(cell_no, cell_info, scope, T=dtype)
        if cell_no == 0:
          cell_node.inputs[0] = input_nodes[0].name
        else:
          cell_node.inputs[0] = new_cell_nodes[scope][-1].name + ":2"
          prev_c_output_shapes = node_dict[cell_info["prev_c"]].attr[
              "_output_shapes"]
          new_cell_nodes[scope][-1].attr[
              "_output_shapes"] = ["", ""] + prev_c_output_shapes

        new_cell_nodes[scope].append(cell_node)
      scope_output_shapes = node_dict[outputs[0]].attr["_output_shapes"]
      new_cell_nodes[scope][-1].attr["_output_shapes"] = scope_output_shapes

      if batch_major:
        perm_node, trans_node = cls._make_major_transpose_nodes(
            outputs, scope, node_dict, new_cell_nodes[scope][-1], True)
        new_cell_nodes[scope].extend([perm_node, trans_node])

      new_cell_nodes[scope][-1].name = outputs[0]

    res_nodes = []
    for g in group_nodes:
      if isinstance(g, list):
        res_nodes.extend(g)
      else:
        res_nodes.extend(new_cell_nodes[g])
    return [
        n if isinstance(n, TensorflowNode) else TensorflowNode(n)
        for n in res_nodes
    ]

  @staticmethod
  def _make_major_transpose_nodes(inputs, scope, node_dict, prev_node, post):
    """Make major transpose nodes if is batch major.

    Args:
      inputs: Inputs names.
      scope: Name scope.
      node_dict: Node dict.
      prev_node: Previous node.
      post: If post transpose flag.

    Returns:
      Perm node.
      Transpose node.

    """
    input_shape = node_dict[inputs[0]].attr["_output_shapes"][0]
    input_rank = len(input_shape)

    perm_node = TensorflowNode()
    perm_node.op_type = "Const"
    perm_node.name = "/".join([scope, "transpose", "perm", get_unique_suffix()])
    perm_node.attr = {
        "value": np.asarray([1, 0] + list(range(input_rank))[2:], np.int32),
        "dtype": data_type.tf2onnx(tf.int32),
        "_output_shapes": [input_rank]
    }

    if post:
      input_shape = [input_shape[i] for i in perm_node.attr["value"]]
      prev_node.attr["_output_shapes"] = [input_shape]

    trans_node = TensorflowNode()
    trans_node.op_type = "Transpose"
    trans_node.name = "/".join([scope, "transpose", get_unique_suffix()])
    trans_node.inputs = [
        inputs[0] if not post else prev_node.name, perm_node.name
    ]
    trans_node.attr["T"] = node_dict[inputs[0]].attr["T"]
    trans_node.attr["_output_shapes"] = [[
        input_shape[i] for i in perm_node.attr["value"]
    ]]
    return [perm_node, trans_node]

  @staticmethod
  def _make_rnn_node(cell_no, cell_info, scope, **kwargs):
    """Make RNN node.

    Args:
      cell_no: Cell No.
      cell_info: Cell info obj.
      scope: Name scope.
      **kwargs: Other args.

    Returns:
      RNN node.

    """
    node = TensorflowNode()
    node.op_type = cell_info["type"].upper()
    node.name = "/".join([
        scope, node.op_type
        if cell_no == 0 else node.op_type + "_{}".format(cell_no)
    ])
    node.inputs = [
        cell_info.get("prev_c", None),
        cell_info.get("kernel", None),
        cell_info.get("bias", None)
    ]
    for k, v in kwargs.items():
      node.attr[k] = v
    return node

  @staticmethod
  def _get_input_output_node_names(nodes):
    """Get input and output node names by given nodes.

    Args:
      nodes:

    Returns:
      Input node names.
      Output node names.
    """
    input_names, output_names = set(), set()
    extension_output_names = set()
    for node in nodes:
      output_names.add(node.name)
      # Add outputs for Split, Switch TensorArrayV3
      if node.op == "Split":
        for i in range(1, node.attr["num_split"].i):
          output_names.add(node.name + ":{}".format(i))
      if node.op == "Switch":
        output_names.add(node.name + ":1")
        extension_output_names.add((node.name, node.name + ":1"))
      if node.op == "TensorArrayV3":
        output_names.add(node.name + ":1")
        extension_output_names.add((node.name, node.name + ":1"))
      input_names.update(set(node.input))
    inputs = input_names - output_names
    outputs = output_names - input_names
    while extension_output_names:
      ext_names = extension_output_names.pop()
      for name in ext_names:
        if name in outputs:
          outputs -= set(ext_names)
          break
    return list(inputs), list(outputs)
