import collections

import numpy as np
import tensorflow as tf

from onnx_tf.common import data_type
from onnx_tf.common import get_unique_suffix
from onnx_tf.pb_wrapper import TensorflowNode


class ScopeParser(object):

  CONST_MINUS_ONE = "_onnx_tf_util_const_minus_one"
  CONST_ZERO = "_onnx_tf_util_const_zero"
  CONST_ONE = "_onnx_tf_util_const_one"

  TRIGGER = {}

  @classmethod
  def parse(cls, nodes):
    for node in nodes:
      for scope in node.name.split("/"):
        if scope in cls.TRIGGER:
          nodes = cls.TRIGGER.pop(scope).parse(nodes)
    return nodes

  @classmethod
  def _add_util_nodes(cls, nodes):
    util_nodes = [("const_minus_one", np.array([-1]).astype(np.int32)),
                  ("const_zero", np.array([0]).astype(np.int32)),
                  ("const_one", np.array([1]).astype(np.int32))]
    for name, value in util_nodes:
      util_node = TensorflowNode(
          op_type="Const",
          name="_onnx_tf_util_{}".format(name),
          attr={
              "value": value,
              "dtype": data_type.any_dtype_to_onnx_dtype(value.dtype),
              "_output_shapes": [value.shape]
          })
      nodes.append(util_node)


class RNNScopeParser(ScopeParser):

  CELL_NAME = ""
  OP_TYPE = ""

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
      if cls.CELL_NAME in scopes:
        idx_rnn_cell = scopes.index(cls.CELL_NAME)
        idx_while = scopes.index(
            "while") if "while" in scopes else len(scopes) - 1
        scope = "/".join(scopes[:min(idx_rnn_cell, idx_while)])
        node_info_holder.scopes.add(scope)
        node_info_holder.cell_dict[
            "type"] = cls.OP_TYPE or scopes[idx_rnn_cell].replace("_cell", "")
        for key in ["kernel", "bias"]:
          if key in scopes[-2:]:
            if key == scopes[-2] and "read" == scopes[-1]:
              node_info_holder.cell_dict[key] = n.name
            node_info_holder.nodes_keep[scope].add(n.name)
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
      if len(scope) > 1:
        raise ValueError(
            "More than one scope {} contained in node name {}.".format(
                str(scope), n.name))
      if scope:
        curr_scope = scope[0]
        if group_nodes[-1]:
          group_nodes.append(curr_scope)
          group_nodes.append([])
        node_info_holder.nodes[curr_scope].append(n)
        if n.name in node_info_holder.nodes_keep[curr_scope]:
          new_cell_nodes[curr_scope].append(n)
      else:
        group_nodes[-1].append(n)
    return group_nodes, new_cell_nodes

  @classmethod
  def add_kernel_and_bias_concat(cls, nodes, node_dict):
    return None, None, None

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

      w_kernel, r_kernel, bias = cls.add_kernel_and_bias_concat(
          new_cell_nodes[scope], node_dict)
      node_info_holder.cell_dict["inputs"] = {
          "prev_c": input_nodes[0].outputs[0],
          "w_kernel": w_kernel,
          "r_kernel": r_kernel,
          "bias": bias,
      }

      batch_major = [
          n for n in node_info_holder.nodes[scope] if inputs[0] in n.input
      ][0].op == "Transpose"

      if batch_major:
        perm_node, trans_node = cls._make_major_transpose_nodes(
            inputs, scope, node_dict, new_cell_nodes[scope][-1], False)
        input_nodes = [trans_node]
        new_cell_nodes[scope].extend([perm_node, trans_node])

      dtype = input_nodes[0].attr["dtype"]
      cell_node = cls._make_rnn_node(
          0, node_info_holder.cell_dict, scope, dtype=dtype)

      new_cell_nodes[scope].append(cell_node)
      scope_output_shapes = node_dict[outputs[0]].attr["_output_shapes"]
      new_cell_nodes[scope][-1].attr["_output_shapes"] = scope_output_shapes

      if batch_major:
        perm_node, trans_node = cls._make_major_transpose_nodes(
            outputs, scope, node_dict, new_cell_nodes[scope][-1], True)
        new_cell_nodes[scope].extend([perm_node, trans_node])

      new_cell_nodes[scope][-1].outputs = [outputs[0]]

    res_nodes = []
    cls._add_util_nodes(res_nodes)
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

    perm_node = TensorflowNode(
        op_type="Const",
        name="/".join([scope, "transpose", "perm",
                       get_unique_suffix()]),
        attr={
            "value": np.asarray([1, 0] + list(range(input_rank))[2:], np.int32),
            "dtype": data_type.tf2onnx(tf.int32),
            "_output_shapes": [input_rank]
        })

    if post:
      input_shape = [input_shape[i] for i in perm_node.attr["value"]]
      prev_node.attr["_output_shapes"] = [input_shape]

    trans_node = TensorflowNode(
        op_type="Transpose",
        name="/".join([scope, "transpose",
                       get_unique_suffix()]),
        inputs=[inputs[0] if not post else prev_node.name, perm_node.name],
        attr={
            "T": node_dict[inputs[0]].attr["T"],
            "_output_shapes":
            [[input_shape[i] for i in perm_node.attr["value"]]]
        })
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
        scope,
        node.op_type if cell_no == 0 else node.op_type + "_{}".format(cell_no)
    ])
    node.inputs = [
        cell_info["inputs"]["prev_c"], cell_info["inputs"]["w_kernel"],
        cell_info["inputs"]["r_kernel"], cell_info["inputs"]["bias"]
    ]
    node.outputs = node.get_outputs_names(num=2)
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


class BasicRNNScopeParser(RNNScopeParser):

  CELL_NAME = "basic_rnn_cell"
  OP_TYPE = "RNN"

  @classmethod
  def add_kernel_and_bias_concat(cls, nodes, node_dict):
    kb_node_dict = {"kernel": {}, "bias": {}}
    scope = ""
    for node in nodes:
      scopes = node.name.split("/")
      for key in ("kernel", "bias"):
        if scopes[-2] == key and scopes[-1] == "read":
          kb_node_dict[key] = node
          if not scope:
            scope = "/".join(scopes[:-3])

    new_kernel = None
    new_bias = None

    for key, value in kb_node_dict.items():
      output_shape = node_dict[value.name].attr["_output_shapes"][0]
      if key == "kernel":
        hidden_size = output_shape[1]
        input_size = output_shape[0] - hidden_size
        transposed_shape = output_shape[::-1]
        transpose_node = TensorflowNode(
            op_type="Transpose",
            name="/".join([scope, key, "transpose_" + get_unique_suffix()]),
            inputs=[value.name, None],
            attr={"_output_shapes": [transposed_shape]})

        split_const_node = TensorflowNode(
            op_type="Const",
            name="/".join([scope, key, "split_const_" + get_unique_suffix()]),
            attr={
                "value": np.asarray([input_size, hidden_size], np.int32),
                "dtype": data_type.tf2onnx(tf.int32),
                "_output_shapes": [[1]]
            })

        split_node = TensorflowNode(
            op_type="SplitV",
            name="/".join([scope, key, "split_" + get_unique_suffix()]),
            inputs=transpose_node.outputs + split_const_node.outputs +
            [cls.CONST_ONE],
            attr={
                "num_split":
                2,
                "_output_shapes": [[hidden_size, input_size],
                                   [hidden_size, hidden_size]]
            })

        nodes.extend([transpose_node, split_const_node, split_node])
        new_kernel = split_node.outputs
      else:
        new_bias = [value.name]
    return new_kernel + new_bias


class GRUScopeParser(RNNScopeParser):

  CELL_NAME = "gru_cell"
  OP_TYPE = "GRU"

  @classmethod
  def add_kernel_and_bias_concat(cls, nodes, node_dict):
    kb_node_dict = {"kernel": {}, "bias": {}}
    scope = ""
    for node in nodes:
      scopes = node.name.split("/")
      for key in ("kernel", "bias"):
        if scopes[-2] == key and scopes[-1] == "read":
          kb_node_dict[key][scopes[-3]] = node
          if not scope:
            scope = "/".join(scopes[:-3])

    new_kernel = None
    new_bias = None

    for key, value in kb_node_dict.items():
      gate_output_shape = node_dict[value["gates"].
                                    name].attr["_output_shapes"][0]
      candidate_output_shape = node_dict[value["candidate"].
                                         name].attr["_output_shapes"][0]
      last_idx = range(len(gate_output_shape))[-1]
      concat_output_shapes = [
          g if i != last_idx else g + c for i, (
              g,
              c) in enumerate(zip(gate_output_shape, candidate_output_shape))
      ]
      concat_node = TensorflowNode(
          op_type="ConcatV2",
          name="/".join([scope, key, "concat_" + get_unique_suffix()]),
          inputs=[
              value["gates"].name, value["candidate"].name, cls.CONST_MINUS_ONE
          ],
          attr={"_output_shapes": [concat_output_shapes]})
      nodes.append(concat_node)

      if key == "kernel":
        hidden_size = gate_output_shape[1] // 2
        input_size = gate_output_shape[0] - hidden_size
        transposed_shape = concat_output_shapes[::-1]
        transpose_node = TensorflowNode(
            op_type="Transpose",
            name="/".join([scope, key, "transpose_" + get_unique_suffix()]),
            inputs=concat_node.outputs + [None],
            attr={"_output_shapes": [transposed_shape]})

        split_const_node = TensorflowNode(
            op_type="Const",
            name="/".join([scope, key, "split_const_" + get_unique_suffix()]),
            attr={
                "value": np.asarray([input_size, hidden_size], np.int32),
                "dtype": data_type.tf2onnx(tf.int32),
                "_output_shapes": [[1]]
            })

        split_node = TensorflowNode(
            op_type="Split",
            name="/".join([scope, key, "split_" + get_unique_suffix()]),
            inputs=[cls.CONST_ZERO] + transpose_node.outputs,
            attr={
                "num_split":
                3,
                "_output_shapes":
                [[int(transposed_shape[0] / 3), transposed_shape[1]]
                 for _ in range(3)]
            })

        re_concat_node = TensorflowNode(
            op_type="ConcatV2",
            name="/".join([scope, key, "re_concat_" + get_unique_suffix()]),
            inputs=[
                split_node.outputs[1], split_node.outputs[0], cls.CONST_ZERO
            ],
            attr={
                "_output_shapes":
                [[int(transposed_shape[0] / 3 * 2), transposed_shape[1]]]
            })

        nodes.extend(
            [transpose_node, split_const_node, split_node, re_concat_node])
        new_kernel = re_concat_node.outputs + [split_node.outputs[2]]
      else:
        new_bias = concat_node.outputs

    return new_kernel + new_bias


class BasicLSTMScopeParser(RNNScopeParser):

  CELL_NAME = "basic_lstm_cell"
  OP_TYPE = "LSTM"

  @classmethod
  def add_kernel_and_bias_concat(cls, nodes, node_dict):
    kb_node_dict = {"kernel": {}, "bias": {}}
    scope = ""
    for node in nodes:
      scopes = node.name.split("/")
      for key in ("kernel", "bias"):
        if scopes[-2] == key and scopes[-1] == "read":
          kb_node_dict[key] = node
          if not scope:
            scope = "/".join(scopes[:-3])

    new_kernel = None
    new_bias = None

    for key, value in kb_node_dict.items():
      output_shape = node_dict[value.name].attr["_output_shapes"][0]
      if key == "kernel":
        hidden_size = output_shape[1] // 4
        input_size = output_shape[0] - hidden_size
        transposed_shape = output_shape[::-1]
        transpose_node = TensorflowNode(
            op_type="Transpose",
            name="/".join([scope, key, "transpose_" + get_unique_suffix()]),
            inputs=[value.name, None],
            attr={"_output_shapes": [transposed_shape]})

        split_node = TensorflowNode(
            op_type="Split",
            name="/".join([scope, key, "split_" + get_unique_suffix()]),
            inputs=[cls.CONST_ZERO] + transpose_node.outputs,
            attr={
                "num_split":
                4,
                "_output_shapes":
                [[hidden_size, input_size + hidden_size] for _ in range(4)]
            })

        concat_node = TensorflowNode(
            op_type="ConcatV2",
            name="/".join([scope, key, "concat_" + get_unique_suffix()]),
            inputs=[
                split_node.outputs[0], split_node.outputs[3],
                split_node.outputs[2], split_node.outputs[1], cls.CONST_ZERO
            ],
            attr={"_output_shapes": [transposed_shape]})

        re_split_const_node = TensorflowNode(
            op_type="Const",
            name="/".join([scope, key,
                           "re_split_const_" + get_unique_suffix()]),
            attr={
                "value": np.asarray([input_size, hidden_size], np.int32),
                "dtype": data_type.tf2onnx(tf.int32),
                "_output_shapes": [[1]]
            })

        re_split_node = TensorflowNode(
            op_type="SplitV",
            name="/".join([scope, key, "re_split_" + get_unique_suffix()]),
            inputs=concat_node.outputs + re_split_const_node.outputs +
            [cls.CONST_ONE],
            attr={
                "num_split":
                2,
                "_output_shapes": [[4 * hidden_size, input_size],
                                   [4 * hidden_size, hidden_size]]
            })

        nodes.extend([
            transpose_node, split_node, concat_node, re_split_const_node,
            re_split_node
        ])
        new_kernel = re_split_node.outputs
      else:
        split_node = TensorflowNode(
            op_type="Split",
            name="/".join([scope, key, "split_" + get_unique_suffix()]),
            inputs=[cls.CONST_ZERO, value.name],
            attr={
                "num_split": 4,
                "_output_shapes": [[int(output_shape[0] / 4)] for _ in range(4)]
            })

        concat_node = TensorflowNode(
            op_type="ConcatV2",
            name="/".join([scope, key, "concat_" + get_unique_suffix()]),
            inputs=[
                split_node.outputs[0], split_node.outputs[3],
                split_node.outputs[2], split_node.outputs[1], cls.CONST_ZERO
            ],
            attr={"_output_shapes": [output_shape]})
        nodes.extend([split_node, concat_node])
        new_bias = concat_node.outputs
    return new_kernel + new_bias


class LSTMScopeParser(BasicLSTMScopeParser):

  CELL_NAME = "lstm_cell"


class MultiRNNScopeParser(RNNScopeParser):

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
        cell_node = cls._make_rnn_node(cell_no, cell_info, scope, dtype=dtype)
        if cell_no == 0:
          cell_node.inputs[0] = input_nodes[0].name
        else:
          cell_node.inputs[0] = new_cell_nodes[scope][-1].name + ":2"
          prev_c_output_shapes = node_dict[
              cell_info["prev_c"]].attr["_output_shapes"]
          new_cell_nodes[scope][-1].attr["_output_shapes"] = [
              "", ""
          ] + prev_c_output_shapes

        new_cell_nodes[scope].append(cell_node)
      scope_output_shapes = node_dict[outputs[0]].attr["_output_shapes"]
      new_cell_nodes[scope][-1].attr["_output_shapes"] = scope_output_shapes

      if batch_major:
        perm_node, trans_node = cls._make_major_transpose_nodes(
            outputs, scope, node_dict, new_cell_nodes[scope][-1], True)
        new_cell_nodes[scope].extend([perm_node, trans_node])

      new_cell_nodes[scope][-1].outputs = [outputs[0]]

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


for parser in (BasicRNNScopeParser, BasicLSTMScopeParser, GRUScopeParser):
  ScopeParser.TRIGGER[parser.CELL_NAME] = parser
