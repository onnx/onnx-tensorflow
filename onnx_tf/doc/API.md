ONNX-Tensorflow API
======

#### `onnx_tf.backend.prepare`

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

#### `onnx_tf.backend_rep.TensorflowRep.export_graph`

<details>
  <summary>Export backend representation to a Tensorflow proto file.

  </summary>
This function obtains the graph proto corresponding to the ONNX
model associated with the backend representation and serializes
to a protobuf file.

</details>



_params_:

`path` : the path to the output TF protobuf file.


_returns_:

none.

#### `onnx_tf.frontend.tensorflow_graph_to_onnx_model`

<details>
  <summary>Converts a Tensorflow Graph Proto to an ONNX model

  </summary>
This function converts a Tensorflow Graph proto to an equivalent
representation of ONNX model.

</details>



_params_:

`graph_def` : Tensorflow Graph Proto object.


`output` : A string specifying the name of the output
graph node.


`opset` : Opset version of the operator set.
Default 0 means using latest version.


`producer_name` : The name of the producer.


`graph_name` : The name of the output ONNX Graph.


_returns_:

The equivalent ONNX Model Proto object.

