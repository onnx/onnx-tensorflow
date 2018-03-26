from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


class TensorflowNet(object):
  """
	Placeholder class for a protobuf definition.
  """

  def __init__(self):
    # Record the computational graph
    self.graph = None
    # Record string names of input tensors as defined in
    # the ONNX model.
    self.external_input = []
    # Record string names of output tensors as defined in
    # the ONNX model.
    self.external_output = []
    # String name -> TF tensor map that records every tensor
    # produced for the execution of the graph.
    self.tensor_dict = {}
