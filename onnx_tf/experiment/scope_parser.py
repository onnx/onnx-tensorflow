import collections
from enum import Enum, auto

import numpy as np
import tensorflow as tf

from onnx_tf.common import data_type
from onnx_tf.common import get_unique_suffix
from onnx_tf.common import CONST_MINUS_ONE_INT32
from onnx_tf.common import CONST_ONE_INT32
from onnx_tf.common import CONST_ZERO_INT32
from onnx_tf.pb_wrapper import TensorflowNode


class RNNType(Enum):
  RNN = auto()
  GRU = auto()
  LSTM = auto()


class ScopeParser(object):

  @classmethod
  def parse(cls, nodes):
    return nodes


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
      self.nodes_keep = set()

  @classmethod
  def _make_node_info(cls, nodes):
    """Make NodeInfoHolder object.

    Args:
      nodes: List of TensorflowNode.

    Returns:
      NodeInfoHolder object.

    """
    node_info_holder = cls.NodeInfoHolder()
    for n in nodes:
      scopes = n.name.split("/")
      for key in ["kernel", "bias"]:
        if key not in scopes:
          continue
        if cls.CELL_NAME in scopes and key == scopes[-2] and "read" == scopes[-1]:
          node_info_holder.cell_dict.setdefault(key, []).append(n.name)
        node_info_holder.nodes_keep.add(n.name)
      if "while" in scopes and cls.CELL_NAME in scopes:
        idx_while = scopes.index("while")
        idx_rnn_cell = scopes.index(cls.CELL_NAME)
        scope = "/".join(scopes[:idx_while])
        node_info_holder.scopes.add(scope)
        node_info_holder.cell_dict[
            "type"] = cls.OP_TYPE or scopes[idx_rnn_cell].replace("_cell", "")
    return node_info_holder

  @classmethod
  def _group_nodes(cls, nodes, node_info_holder):
    """Grouping nodes into [nodes, cell_nodes_key, nodes].

    Args:
      nodes: List of TensorflowNode.
      node_info_holder: NodeInfoHolder object.

    Returns:
      Grouped nodes list.
      Cell nodes dict.

    """
    group_nodes = [[]]
    kb_nodes = []
    for n in nodes:
      matched_scope = []
      if n.name in node_info_holder.nodes_keep:
        kb_nodes.append(n)
        continue
      for scope in node_info_holder.scopes:
        if scope in n.name and n.name[n.name.find(scope) + len(scope)] == "/":
          matched_scope.append(scope)
      if len(matched_scope) > 1:
        raise ValueError(
            "More than one scope {} contained in node name {}.".format(
                str(matched_scope), n.name))
      if matched_scope:
        curr_scope = matched_scope[0]
        if group_nodes[-1]:
          group_nodes.append(curr_scope)
          group_nodes.append([])
        node_info_holder.nodes[curr_scope].append(n)
      else:
        group_nodes[-1].append(n)
    return group_nodes, kb_nodes

  @classmethod
  def process_kernel_and_bias(cls, nodes, cell_dict, node_dict):
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
    node_dict = {n.name: n for n in nodes}
    group_nodes, kb_nodes = cls._group_nodes(nodes, node_info_holder)

    new_cell_nodes = collections.defaultdict(list)

    for scope in node_info_holder.nodes:
      inputs, outputs = cls._get_input_output_node_names(
          node_info_holder.nodes[scope])
      inputs = [i for i in inputs if scope not in i]

      sorted_inputs = ["", "", ""]
      for inp in inputs:
        if "kernel" in inp:
          sorted_inputs[1] = inp
        elif "bias" in inp:
          sorted_inputs[2] = inp
        else:
          sorted_inputs[0] = inp
      input_nodes = [node_dict[i] for i in sorted_inputs if i in node_dict]

      w_kernel, r_kernel, bias = cls.process_kernel_and_bias(
          new_cell_nodes[scope], node_info_holder.cell_dict, node_dict)

      batch_major = [
          n for n in node_info_holder.nodes[scope] if inputs[0] in n.inputs
      ][0].op_type == "Transpose"

      if batch_major:
        perm_node, trans_node = cls._make_major_transpose_nodes(
            sorted_inputs, scope, node_dict, new_cell_nodes[scope][-1], False)
        input_nodes[0] = trans_node
        new_cell_nodes[scope].extend([perm_node, trans_node])

      node_info_holder.cell_dict["inputs"] = {
          "prev_c": input_nodes[0].outputs[0],
          "w_kernel": w_kernel,
          "r_kernel": r_kernel,
          "bias": bias,
      }

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

      new_cell_nodes[scope][-1].outputs = [
          output.replace(new_cell_nodes[scope][-1].name, outputs[0])
          for output in new_cell_nodes[scope][-1].outputs
      ]
      new_cell_nodes[scope][-1].name = outputs[0]

    res_nodes = kb_nodes
    for g in group_nodes:
      if isinstance(g, list):
        res_nodes.extend(g)
      else:
        res_nodes.extend(new_cell_nodes[g])
    return res_nodes

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
            "dtype": data_type.tf2onnx(node_dict[inputs[0]].attr["T"]),
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
        scope, node.op_type
        if cell_no == 0 else node.op_type + "_{}".format(cell_no)
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
      tf_node = node if isinstance(node,
                                   TensorflowNode) else TensorflowNode(node)
      output_names.add(node.name)
      # Add outputs for Split, Switch TensorArrayV3
      if tf_node.op_type == "Split":
        for i in range(1, tf_node.attr["num_split"]):
          output_names.add(tf_node.name + ":{}".format(i))
      if tf_node.op_type == "Switch":
        output_names.add(tf_node.name + ":1")
        extension_output_names.add((tf_node.name, tf_node.name + ":1"))
      if tf_node.op_type == "TensorArrayV3":
        output_names.add(tf_node.name + ":1")
        extension_output_names.add((tf_node.name, tf_node.name + ":1"))
      input_names.update(
          set([inp if inp[0] != "^" else inp[1:] for inp in tf_node.inputs]))
    inputs = input_names - output_names
    outputs = output_names - input_names
    while extension_output_names:
      ext_names = extension_output_names.pop()
      for name in ext_names:
        if name in outputs:
          outputs -= set(ext_names)
          break
    inputs.discard(None)
    return list(inputs), list(outputs)


class BasicRNNScopeParser(RNNScopeParser):

  CELL_NAME = "basic_rnn_cell"
  OP_TYPE = "RNN"

  @classmethod
  def process_kernel_and_bias(cls, nodes, cell_dict, node_dict):
    new_kernel = None
    new_bias = None
    scopes = cell_dict["kernel"].split("/")
    scope = "/".join(scopes[:scopes.index("kernel")])
    for key, value in [("kernel", node_dict[cell_dict["kernel"][0]]),
                       ("bias", node_dict[cell_dict["bias"][0]])]:
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
            [CONST_ONE_INT32],
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
  def process_kernel_and_bias(cls, nodes, cell_dict, node_dict):
    new_kernel = None
    new_bias = None
    scopes = cell_dict["kernel"][0].split("/")
    scope = "/".join(scopes[:scopes.index("kernel")])
    for key, value in [[
        "kernel", [node_dict[kernel] for kernel in cell_dict["kernel"]]
    ], ["bias", [node_dict[bias] for bias in cell_dict["bias"]]]]:
      gate_output_shape = node_dict[value[0].name].attr["_output_shapes"][0]
      candidate_output_shape = node_dict[value[1].name].attr["_output_shapes"][
          0]
      last_idx = range(len(gate_output_shape))[-1]
      concat_output_shapes = [
          g if i != last_idx else g + c for i, (
              g,
              c) in enumerate(zip(gate_output_shape, candidate_output_shape))
      ]
      concat_node = TensorflowNode(
          op_type="ConcatV2",
          name="/".join([scope, key, "concat_" + get_unique_suffix()]),
          inputs=[value[0].name, value[1].name, CONST_MINUS_ONE_INT32],
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
            inputs=[CONST_ZERO_INT32] + transpose_node.outputs,
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
                split_node.outputs[1], split_node.outputs[0], CONST_ZERO_INT32
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
  def process_kernel_and_bias(cls, nodes, cell_dict, node_dict):
    new_kernel = None
    new_bias = None
    scopes = cell_dict["kernel"].split("/")
    scope = "/".join(scopes[:scopes.index("kernel")])
    for key, value in [("kernel", node_dict[cell_dict["kernel"][0]]),
                       ("bias", node_dict[cell_dict["bias"][0]])]:
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
            inputs=[CONST_ZERO_INT32] + transpose_node.outputs,
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
                split_node.outputs[2], split_node.outputs[1], CONST_ZERO_INT32
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
            [CONST_ONE_INT32],
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
            inputs=[CONST_ZERO_INT32, value.name],
            attr={
                "num_split": 4,
                "_output_shapes": [[int(output_shape[0] / 4)] for _ in range(4)]
            })

        concat_node = TensorflowNode(
            op_type="ConcatV2",
            name="/".join([scope, key, "concat_" + get_unique_suffix()]),
            inputs=[
                split_node.outputs[0], split_node.outputs[3],
                split_node.outputs[2], split_node.outputs[1], CONST_ZERO_INT32
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
              node_info_holder.cell_dict[cell_no].setdefault(key,
                                                             []).append(n.name)
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
    node_dict = {
        n.name: TensorflowNode(n) if not isinstance(n, TensorflowNode) else n
        for n in nodes
    }
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


def get_rnn_scope_parser(rnn_type):
  if isinstance(rnn_type, str):
    rnn_type = RNNType[rnn_type]
  if rnn_type == RNNType.RNN:
    return BasicRNNScopeParser
  elif rnn_type == RNNType.GRU:
    return GRUScopeParser
  elif rnn_type == RNNType.LSTM:
    return BasicLSTMScopeParser
  else:
    return None
