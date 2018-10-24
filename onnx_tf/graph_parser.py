class Parser(object):

  @classmethod
  def parse(cls, graph_def):
    return graph_def


def get_input_output_node_names(nodes):
  input_names, output_names= set(), set()
  extension_output_names = set()
  for node in nodes:
    output_names.add(node.name)
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


class MultiRNNParser(Parser):

  class NodeInfoHolder(object):

    def __init__(self):
      self.scopes = set()
      self.nodes = dict()
      self.cell_dict = dict()

  @classmethod
  def parse(cls, graph_def):
    node_info_holder = cls.NodeInfoHolder()
    for n in graph_def.node:
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
        node_info_holder.cell_dict.setdefault(cell_no, {})["type"] = scopes[idx_cell_type]
        if "read" in scopes:
          if "kernel" in scopes:
            node_info_holder.cell_dict[cell_no]["kernel"] = n
          if "bias" in scopes:
            node_info_holder.cell_dict[cell_no]["bias"] = n

    for n in graph_def.node:
      for s in node_info_holder.scopes:
        if s in n.name:
          node_info_holder.nodes.setdefault(s, []).append(n)
    for k in node_info_holder.nodes:
      inputs, outputs = get_input_output_node_names(node_info_holder.nodes[k])
      print(inputs)
      print(outputs)
    return graph_def
