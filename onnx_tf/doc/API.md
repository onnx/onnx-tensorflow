ONNX-Tensorflow API
------

#### `onnx_tf.frontend.TensorflowFrontendBase.tensorflow_graph_to_onnx_graph`

Converts a Tensorflow Graph Proto to an ONNX graph

This function converts a Tensorflow Graph proto to an equivalent
representation of ONNX graph.

_params_:

`graph_def` : Tensorflow Graph Proto object.


`output` : A Tensorflow NodeDef object specifying which node
to be taken as output of the ONNX graph.


`opset` : Opset version of the operator set.
Default 0 means using latest version.


`name` : The name of the output ONNX Graph.


#### `onnx_tf.backend.TensorflowBackendBase.prepare`

Prepare an ONNX model for Tensorflow Backend

This function converts an ONNX model to an internel representation
of the computational graph called TensorflowRep and returns
the converted representation.

_params_:

`model` : the ONNX model to be converted


`device` : the device to execute this model on


