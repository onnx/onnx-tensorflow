class Parser(object):

  @classmethod
  def parse(cls, graph_def):
    return graph_def


def get_input_output_node_names(nodes):
  nodes_dict, input_names = dict(), set()
  for node in nodes:
    nodes_dict[node.name] = node
    input_names.update(set(node.input))
  return list(input_names - set(nodes_dict)), list(set(nodes_dict) - input_names)


class MultiRNNPaser(Parser):

  @classmethod
  def parse(cls, graph_def):
    rnn_scopes = dict()
    for n in graph_def.node:
      scopes = n.name.split("/")
      if "multi_rnn_cell" in scopes:
        idx_multi_rnn_cell = scopes.index("multi_rnn_cell")
        idx_while = scopes.index("while") if "while" in scopes else len(scopes) - 1
        scope = "/".join(scopes[:min(idx_multi_rnn_cell, idx_while)])
        rnn_scopes.setdefault(scope, []).append(n)
        # for n in graph_def.node:
        print(scopes, scope)
    get_input_output_node_names(list(rnn_scopes.values())[0])
    return graph_def
