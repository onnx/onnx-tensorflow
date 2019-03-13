ONNX-Tensorflow Command Line Interface
======

## Available commands:
- convert

More information: `onnx-tf -h`
```

WARNING: The TensorFlow contrib module will not be included in TensorFlow 2.0.
For more information, please see:
  * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
  * https://github.com/tensorflow/addons
If you depend on functionality not listed there, please file an issue.

usage: onnx-tf [-h] {convert,check,optimize}

ONNX-Tensorflow Command Line Interface

positional arguments:
  {convert,check,optimize}
                        Available commands.

optional arguments:
  -h, --help            show this help message and exit
```

## Usage:

### Convert:

#### From Tensorflow to ONNX:

- Use frozen pb:
`onnx-tf convert -t onnx -i /path/to/input.pb -o /path/to/output.onnx`

- Use ckpt:
`onnx-tf convert -t onnx -i /path/to/input.ckpt -o /path/to/output.onnx`
(`/path/to` folder should contain files: `checkpoint`, `*.ckpt-{step}.data-*`, `*.ckpt-{step}.index`, `.ckpt-{step}.meta`)

#### From ONNX to Tensorflow:
`onnx-tf convert -t tf -i /path/to/input.onnx -o /path/to/output.pb`

More information: `onnx-tf convert -h`
```

WARNING: The TensorFlow contrib module will not be included in TensorFlow 2.0.
For more information, please see:
  * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
  * https://github.com/tensorflow/addons
If you depend on functionality not listed there, please file an issue.

usage: onnx-tf [-h] --infile INFILE --outfile OUTFILE --convert_to {onnx,tf}
               [--graph GRAPH] [--device DEVICE] [--strict STRICT]
               [--output OUTPUT] [--opset OPSET]
               [--ignore_unimplemented IGNORE_UNIMPLEMENTED]
               [--optimizer_passes OPTIMIZER_PASSES]
               [--rnn_type {GRU,LSTM,RNN}]

This is the converter for converting protocol buffer between tf and onnx.

optional arguments:
  -h, --help            show this help message and exit
  --infile INFILE, -i INFILE
                        Input file path, can be pb or ckpt file.
  --outfile OUTFILE, -o OUTFILE
                        Output file path.
  --convert_to {onnx,tf}, -t {onnx,tf}
                        Format converted to.
  --graph GRAPH, -g GRAPH
                        Inference graph, which is obtained by optimizing or
                        editing the training graph for better training
                        usability.

backend arguments (onnx -> tf):
  --device DEVICE       The device to execute this model on. (from
                        onnx_tf.backend.prepare)
  --strict STRICT       Whether to enforce semantic equivalence between the
                        original model and the converted tensorflow model,
                        defaults to True (yes, enforce semantic equivalence).
                        Changing to False is strongly discouraged. Currently,
                        the strict flag only affects the behavior of MaxPool
                        and AveragePool ops. (from onnx_tf.backend.prepare)

frontend arguments (tf -> onnx):
  --output OUTPUT       List of string or a string specifying the name of the
                        output graph node. (from
                        onnx_tf.frontend.tensorflow_graph_to_onnx_model)
  --opset OPSET         Opset version number, list or tuple. Default is 0
                        means using latest version with domain ''. List or
                        tuple items should be (str domain, int version
                        number). (from
                        onnx_tf.frontend.tensorflow_graph_to_onnx_model)
  --ignore_unimplemented IGNORE_UNIMPLEMENTED
                        Convert to ONNX model and ignore all the operators
                        that are not currently supported by onnx-tensorflow.
                        This is an experimental feature. By enabling this
                        feature, the model would not be guaranteed to match
                        the ONNX specifications. (from
                        onnx_tf.frontend.tensorflow_graph_to_onnx_model)
  --optimizer_passes OPTIMIZER_PASSES
                        List of optimization names c.f. https://github.com/onn
                        x/onnx/blob/master/onnx/optimizer.py for available
                        optimization passes. (from
                        onnx_tf.frontend.tensorflow_graph_to_onnx_model)

EXPERIMENTAL ARGUMENTS:
  --rnn_type {GRU,LSTM,RNN}
                        RNN graph type if using experimental feature: convert
                        rnn graph to onnx.
```
