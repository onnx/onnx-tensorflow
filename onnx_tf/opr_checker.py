import argparse
import warnings
import logging
import os

from google.protobuf import text_format
from onnx import defs
import tensorflow as tf

from onnx_tf.common import get_output_node_names
from onnx_tf.common.handler_helper import get_all_frontend_handlers
from onnx_tf.common.handler_helper import get_frontend_coverage
from onnx_tf.pb_wrapper import TensorflowNode

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


def parse_args(args):
  parser = argparse.ArgumentParser(
      description=
      'This is the checker to check whether model graph '\
      'operators are supported by the frontend handlers'
  )
  parser.add_argument(
      "--infile",
      "-i",
      help="Input file path, can be pb, pbtxt, or ckpt file.",
      required=True)

  return parser.parse_args(args)


def check_opr_support(graph_def):
  """ Check support for operators in graph

  :param graph_def: the graph of operations
  :return: whether all operators are supported, supported operators in graph
  """

  logger.info('Checking for unsupported operators...')
  node_dict = set()
  unsupported = set()
  supported = set()
  for node in graph_def.node:
    node_dict.add(node.op)

  node_dict.discard('Placeholder')
  node_dict.discard('Const')

  logger.info('There are %s unique operators in the model file.',
              str(len(node_dict)))

  frontend_coverage, frontend_tf_coverage = get_frontend_coverage()
  frontend_tf_opset_dict = frontend_tf_coverage.get(defs.ONNX_DOMAIN, {})
  for k in node_dict:
    if k not in frontend_tf_opset_dict:
      unsupported.add(k)
    else:
      supported.add(k)

  if unsupported:
    logger.info(
        'There are %s operators currently not supported by the '\
        'ONNX-Tensorflow frontend for your model.',
        str(len(unsupported)))
    logger.info(unsupported)
    return False, supported
  logger.info('All operators in the model are supported!')
  return True, supported


def check_node_args(graph_def, supported):
  """ Check for required node arguments in graph

  :param graph_def: the graph of operations
  :param supported: the supported operators in graph
  :return: whether all required parameters are provided
  """

  logger.info('Checking for required node arguments...')

  opset_dict = {}
  opset_dict[defs.ONNX_DOMAIN] = defs.onnx_opset_version()
  handlers = get_all_frontend_handlers(opset_dict)

  total_nodes = 0
  failed_nodes = 0
  for node in graph_def.node:
    if node.op in supported:
      total_nodes += 1
      tf_node = TensorflowNode(node)
      kwargs = {}
      for inp in node.input:
        for attr_node in graph_def.node:
          if inp == attr_node.name:
            kwargs[inp] = attr_node.attr['value']
            break
      handler = handlers.get(defs.ONNX_DOMAIN, {}).get(node.op, None)
      try:
        handler.args_check(tf_node, consts=kwargs)
      except Exception as e:
        logger.info(e)
        failed_nodes += 1

  logger.info('We checked %d supported nodes for required arguments.',
              total_nodes)
  logger.info('  # of nodes passed the args check: %d',
              total_nodes - failed_nodes)
  logger.info('  # of nodes failed the args check: %d', failed_nodes)
  return failed_nodes == 0


def load_graph_from_ckpt(ckpt_file):
  """ Load graph from a checkpoint

  :param ckpt_file: the checkpoint file
  :return: graph of operations extracted from the checkpoint 
  """

  latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(ckpt_file))
  saver = tf.train.import_meta_graph(latest_ckpt + ".meta")
  with tf.Session() as sess:
    sess.run(
        [tf.global_variables_initializer(),
         tf.local_variables_initializer()])
    saver.restore(sess, latest_ckpt)
    output_node_names = get_output_node_names(sess.graph.as_graph_def())
    graph_def = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(add_shapes=True), output_node_names)
  return graph_def


def check(graphfile):
  ext = os.path.splitext(graphfile)[1]
  graph_def = tf.GraphDef()
  if ext == ".pb":
    with tf.gfile.GFile(graphfile, "rb") as f:
      graph_def.ParseFromString(f.read())
  elif ext == ".pbtxt":
    with tf.gfile.GFile(graphfile, "rb") as f:
      text_format.Merge(f.read(), graph_def)
  elif ext == ".ckpt":
    graph_def = load_graph_from_ckpt(graphfile)
  else:
    raise ValueError(
        'Input file is not supported. Should be .pb, .pbtxt, or .ckpt, '\
        'but get {}.'.format(ext))
  op_passed, supported = check_opr_support(graph_def)
  args_passed = check_node_args(graph_def, supported)
  if op_passed and args_passed:
    logger.info("Your model is good to go!")
  else:
    logger.info("Some work is needed before we can export your model to ONNX.")


def main(args):
  args = parse_args(args)
  check(args.infile)
