ONNX-Tensorflow API
------

#### `onnx_tf.frontend.TensorflowFrontendBase.tensorflow_graph_to_onnx_graph`

<details>
  <summary>Converts a Tensorflow Graph Proto to an ONNX graph

  </summary>
This function converts a Tensorflow Graph proto to an equivalent
representation of ONNX graph.

</details>

_params_:

`graph_def` : Tensorflow Graph Proto object.


`output` : A Tensorflow NodeDef object specifying which node
to be taken as output of the ONNX graph.


`opset` : Opset version of the operator set.
Default 0 means using latest version.


`name` : The name of the output ONNX Graph.


_returns_:

The equivalent ONNX Graph Proto object.

#### `onnx_tf.backend.TensorflowBackendBase.prepare`

<details>
  <summary>Prepare an ONNX model for Tensorflow Backend

  </summary>
This function converts an ONNX model to an internel representation
of the computational graph called TensorflowRep and returns
the converted representation.

</details>

_params_:

`model` : the ONNX model to be converted


`device` : the device to execute this model on


_returns_:

a TensorflowRep class object representing the ONNX model

