from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
from onnx_caffe2.backend import prepare as prepare2
from onnx import helper
from onnx.onnx_pb2 import TensorProto

def find_between(s, first, last):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

class TestLargeModel(unittest.TestCase):
  MODEL_PATH = "../../onnx_models/"

  def test(self):
    _model = onnx.load(self.MODEL_PATH + "shufflenet/model.pb")
    node_count = len(_model.graph.node)
    more_outputs = []
    output_to_check = []
    for node in _model.graph.node:
      more_outputs.append(helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, (100, 100)))
      output_to_check.append(node.output[0])
    _model.graph.output.extend(more_outputs)

    tf_rep = prepare(_model)
    cf_rep = prepare2(_model)

    sample = np.load(self.MODEL_PATH + "shufflenet/test_data_{}.npz".format(str(1)), encoding='bytes')
    inputs = list(sample['inputs'])
    outputs = list(sample['outputs'])

    my_out = tf_rep.run(inputs)
    cf_out = cf_rep.run(inputs)

    for op in output_to_check:
      try:
        np.savetxt(op.replace("/", "__") + ".cf", cf_out[op].flatten(), delimiter='\t')
        np.savetxt(op.replace("/", "__") + ".tf", my_out[op].flatten(), delimiter='\t')
        np.testing.assert_allclose(my_out[op], cf_out[op], rtol=1e-2)
        print(op, "results of this layer are correct within tolerence.")
      except Exception as e:
        np.set_printoptions(threshold=np.inf)
        mismatch_percent = (find_between(str(e), "(mismatch", "%)"))
        print(op, "mismatch with percentage {} %".format(mismatch_percent))

if __name__ == '__main__':
  unittest.main()
  pass