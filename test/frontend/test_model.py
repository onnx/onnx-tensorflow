from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import yaml
import sys, os, tempfile
import zipfile
import logging
import subprocess
if sys.version_info >= (3,):
  import urllib.request as urllib2
  import urllib.parse as urlparse
else:
  import urllib2
  import urlparse

import numpy as np
import tensorflow as tf
import onnx
from tensorflow.python.tools import freeze_graph

from onnx_tf.frontend import tensorflow_graph_to_onnx_model
from onnx_tf.backend import prepare
from onnx_tf.common import supports_device


def get_rnd(shape, low=-1.0, high=1.0, dtype=np.float32, seed=42):
  np.random.seed(seed)
  if dtype == np.float32:
    return (np.random.uniform(low, high,
                              np.prod(shape)).reshape(shape).astype(np.float32))
  elif dtype == np.int32:
    return (np.random.uniform(low, high,
                              np.prod(shape)).reshape(shape).astype(np.int32))
  elif dtype == np.bool_:
    return np.random.choice(a=[False, True], size=shape)


def download_and_extract(url, dest=None):
  u = urllib2.urlopen(url)

  scheme, netloc, path, query, fragment = urlparse.urlsplit(url)
  filename = os.path.basename(path)
  if not filename:
    filename = 'downloaded.file'
  if dest:
    if not os.path.exists(dest):
      os.makedirs(dest)
    filename = os.path.join(dest, filename)

  with open(filename, 'wb') as f:
    meta = u.info()
    meta_func = meta.getheaders if hasattr(meta, 'getheaders') else meta.get_all
    meta_length = meta_func("Content-Length")
    file_size = None
    if meta_length:
      file_size = int(meta_length[0])
    print("Downloading: {0} Bytes: {1}".format(url, file_size))

    file_size_dl = 0
    block_sz = 8192
    while True:
      buffer = u.read(block_sz)
      if not buffer:
        break

      file_size_dl += len(buffer)
      f.write(buffer)

      status = "{0:16}".format(file_size_dl)
      if file_size:
        status += "   [{0:6.2f}%]".format(file_size_dl * 100 / file_size)
      status += chr(13)
    print()

  zip = zipfile.ZipFile(filename)
  zip.extractall(dest)


class TestModel(unittest.TestCase):
  """ Tests for models.
  Tests are dynamically added.
  Therefore edit test_model.yaml to add more tests.
  """
  pass


def create_test(test_model, interface):

  def do_test_expected(self):
    tf.reset_default_graph()
    work_dir = "".join([test_model["name"], "-", "workspace"])
    work_dir_prefix = work_dir + "/"
    download_and_extract(test_model["asset_url"], work_dir)

    if "frozen_path" in test_model:
      frozen_path = work_dir_prefix + test_model["frozen_path"]
    else:
      # Parse metagraph def to obtain graph_def
      meta_graph_def_file = open(work_dir_prefix + test_model["metagraph_path"],
                                 "rb")

      meta_graph_def = tf.MetaGraphDef()
      meta_graph_def.ParseFromString(meta_graph_def_file.read())

      with open(work_dir_prefix + "graph_def.pb", 'wb') as f:
        f.write(meta_graph_def.graph_def.SerializeToString())

      # Proceed to freeze graph:
      freeze_graph.freeze_graph(input_graph=work_dir_prefix + "graph_def.pb",
                                input_saver="",
                                input_binary=True,
                                input_checkpoint=work_dir_prefix + test_model["checkpoint_path"],
                                output_node_names=",".join(test_model["outputs"]),
                                restore_op_name="",
                                filename_tensor_name="",
                                output_graph=work_dir_prefix + "frozen_graph.pb",
                                clear_devices=True,
                                initializer_nodes="")
      # Set the frozen graph path.
      frozen_path = work_dir_prefix + "frozen_graph.pb"

    # Now read the frozen graph and import it:
    with tf.gfile.GFile(frozen_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
      tf.import_graph_def(
          graph_def,
          input_map=None,
          return_elements=None,
          name="",
          producer_op_list=None)

    # Tensorflow feed dict is keyed by tensor.
    tf_feed_dict = {}
    # Backend feed dict is keyed by tensor names.
    backend_feed_dict = {}
    for name, shape in test_model["inputs"].items():
      x_val = get_rnd(shape)
      tf_feed_dict[graph.get_tensor_by_name(name + ":0")] = x_val
      backend_feed_dict[name] = x_val

    tf_output_tensors = []
    backend_output_names = []
    for name in test_model["outputs"]:
      tf_output_tensors.append(graph.get_tensor_by_name(name + ":0"))
      backend_output_names.append(name)

    # Obtain reference tensorflow outputs:
    with tf.Session(graph=graph) as sess:
      logging.debug("ops in the graph:")
      logging.debug(graph.get_operations())
      output_tf = sess.run(tf_output_tensors, feed_dict=tf_feed_dict)

    # Now we convert tensorflow models to onnx,
    # if we use python API for conversion:
    if interface == "python":
      model = tensorflow_graph_to_onnx_model(graph_def, backend_output_names, ignore_unimplemented=True)
    else:  # else we use CLI utility
      assert interface == "cli"
      subprocess.check_call([
          "onnx-tf",
          "convert",
          "-t",
          "onnx",
          "-i",
          work_dir_prefix + "frozen_graph.pb",
          "-o",
          work_dir_prefix + "model.onnx",
      ])
      model = onnx.load_model(work_dir_prefix + "model.onnx")

    # Run the output onnx models in our backend:
    tf_rep = prepare(model)
    output_onnx_tf = tf_rep.run(backend_feed_dict)

    assert len(output_tf) == len(output_onnx_tf)
    for tf_output, onnx_backend_output in zip(output_tf, output_onnx_tf):
      np.testing.assert_allclose(
          tf_output, onnx_backend_output, rtol=1e-3, atol=1e-7)

  return do_test_expected


dir_path = os.path.dirname(os.path.realpath(__file__))
with open(dir_path + "/test_model.yaml", 'r') as config:
  try:
    for test_model in yaml.safe_load_all(config):
      for device in test_model["devices"]:
        for interface in ["python", "cli"]:
          if supports_device(device):
            test_method = create_test(test_model, interface)
            test_name_parts = ["test", test_model["name"], device, interface]
            test_name = str("_".join(map(str, test_name_parts)))
            test_method.__name__ = test_name
            setattr(TestModel, test_method.__name__, test_method)
  except yaml.YAMLError as exception:
    print(exception)

if __name__ == '__main__':
  unittest.main()
