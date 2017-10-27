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