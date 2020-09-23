from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np

import onnx
from onnx import helper
from onnx import TensorProto
import tensorflow as tf
import onnxruntime.backend as ort

import onnx_tf.backend as otf
from onnx_tf.common import data_type


def find_between(s, first, last):
  try:
    start = s.index(first)
    end = s.index(last) + len(last)
    return s[start:end]
  except ValueError:
    return ""


class TestMnistModel(unittest.TestCase):
  # Make sure the onnx file path is correct, assuming copied to the
  # current directory
  model_path = 'mnist-8.onnx'

  def test(self):
    _model = onnx.load(self.model_path)
    print("Total node count in model: ", len(_model.graph.node))

    # The input tensors could be provided as constants
    # The example below illustrates such a dictionary could be
    # provided for models with unknown input shapes. Since
    # mnist has known input shape, we don't provide input tensors.
    # input_tensors = {'Input3': tf.constant(0, dtype = tf.float32,
    #                    name='Input3',
    #                    shape=[1, 1, 28, 28])}
    input_tensors = {}
    tensor_dict = otf.prepare(_model,
                              gen_tensor_dict=True,
                              input_tensor_dict=input_tensors).tensor_dict
    more_outputs = []
    output_to_check = []
    for node in _model.graph.node:
      # add the first output of each node to the model output
      output_tensor = None
      for i in range(len(_model.graph.value_info)):
        if _model.graph.value_info[i].name == node.output[0]:
          output_tensor = _model.graph.value_info[i]

      for i in range(len(_model.graph.initializer)):
        if _model.graph.initializer[i].name == node.output[0]:
          output_tensor = _model.graph.initializer[i]

      # assume the first output is a tensor
      tensor = tensor_dict[node.output[0]]
      output_tensor = helper.make_tensor_value_info(
          node.output[0], data_type.tf2onnx(tensor.dtype),
          tensor.shape) if output_tensor is None else output_tensor
      more_outputs.append(output_tensor)
      output_to_check.append(node.output[0])
    _model.graph.output.extend(more_outputs)

    tf_rep = otf.prepare(_model)
    rt_rep = ort.prepare(_model)

    # prepare input data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    sample = x_test[:1].reshape(1, 1, 28, 28).astype(np.float32)

    inputs = [sample]
    my_out = tf_rep.run(inputs)
    rt_out = rt_rep.run(inputs)

    for op in output_to_check:
      for i in range(len(my_out)):
        # find the index of output in the list
        if my_out[op] is my_out[i]:

          try:
            np.savetxt(op.replace("/", "__") + ".rt",
                       rt_out[i].flatten(),
                       delimiter='\t')
            np.savetxt(op.replace("/", "__") + ".tf",
                       my_out[i].flatten(),
                       delimiter='\t')
            np.testing.assert_allclose(my_out[i], rt_out[i], rtol=1e-2)
            print(op, "results of this layer are correct within tolerence.")
          except Exception as e:
            np.set_printoptions(threshold=np.inf)
            mismatch_percent = (find_between(str(e), "(mismatch", "%)"))
            print(op, "mismatch with percentage {} %".format(mismatch_percent))


if __name__ == '__main__':
  unittest.main()
  pass
