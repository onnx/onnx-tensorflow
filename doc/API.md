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

#### `onnx_tf.frontend.tensorflow_graph_to_onnx_model`

<details>
  <summary>Converts a Tensorflow Graph Proto to an ONNX model

  </summary>
This function converts a Tensorflow Graph proto to an equivalent
representation of ONNX model.

</details>



_params_:

`graph_def` : Tensorflow Graph Proto object.


`output` : List of string or a string specifying the name
of the output graph node.


`opset` : Opset version number, list or tuple.
Default is 0 means using latest version with domain ''.
List or tuple items should be (str domain, int version number).


`producer_name` : The name of the producer.


`graph_name` : The name of the output ONNX Graph.


`ignore_unimplemented` : Convert to ONNX model and ignore all the operators
that are not currently supported by onnx-tensorflow.
This is an experimental feature. By enabling this feature,
the model would not be guaranteed to match the ONNX specifications.


`optimizer_passes` : List of optimization names c.f.
https://github.com/onnx/onnx/blob/master/onnx/optimizer.py for available
optimization passes.


_returns_:

The equivalent ONNX Model Proto object.

