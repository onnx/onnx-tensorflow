ONNX-Tensorflow API
======

#### `onnx_tf.backend.prepare`

<details>
  <summary>Prepare an ONNX model for Tensorflow Backend.

  </summary>
This function converts an ONNX model to an internel representation
of the computational graph called TensorflowRep and returns
the converted representation.

</details>



_params_:

`model` : The ONNX model to be converted.


`device` : The device to execute this model on.


`strict` : Whether to enforce semantic equivalence between the original model
and the converted tensorflow model, defaults to True (yes, enforce semantic equivalence).
Changing to False is strongly discouraged.
Currently, the strict flag only affects the behavior of MaxPool and AveragePool ops.


`logging_level` : The logging level, default is INFO. Change it to DEBUG
to see more conversion details or to WARNING to see less


_returns_:

A TensorflowRep class object representing the ONNX model

#### `onnx_tf.backend_rep.TensorflowRep.export_graph`

<details>
  <summary>Export backend representation to a Tensorflow proto file.

  </summary>
This function obtains the graph proto corresponding to the ONNX
model associated with the backend representation and serializes
to a protobuf file.

</details>



_params_:

`path` : The path to the output TF protobuf file.


_returns_:

none.

