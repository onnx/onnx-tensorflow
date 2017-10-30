from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

class TensorflowNet(object):
  """
	Placeholder class for a protobuf definition.
  """
  def __init__(self):
  	self.op = []
  	self.external_input = []
  	self.external_output = []
  	self.output = []

  	self.output_dict = {}