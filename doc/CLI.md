ONNX-Tensorflow Command Line Interface
======

## Available commands:
- convert

More information: `onnx-tf -h`
```
usage: onnx-tf [-h] {convert}

onnx-tensorflow command line interface

positional arguments:
  {convert}   Available commands.

optional arguments:
  -h, --help  show this help message and exit
```

## Usage:

### Convert:

#### From Tensorflow to ONNX:

- Use frozen pb:
`onnx-tf convert -t onnx -i /path/to/input.pb -o /path/to/output.onnx --ignore_unimplemented True`

- Use ckpt:
`onnx-tf convert -t onnx -i /path/to/input.ckpt -o /path/to/output.onnx --ignore_unimplemented True`
(`/path/to` folder should contain files: `checkpoint`, `*.ckpt-{step}.data-*`, `*.ckpt-{step}.index`, `.ckpt-{step}.meta`)

#### From ONNX to Tensorflow:
`onnx-tf convert -t tf -i /path/to/input.onnx -o /path/to/output.pb`

More information: `onnx-tf convert -h`
```
usage: onnx-tf [-h] --infile INFILE --outfile OUTFILE --convert_to {onnx,tf}
               [--device DEVICE] [--strict STRICT] [--opset OPSET]
               [--ignore_unimplemented IGNORE_UNIMPLEMENTED]
               [--optimizer_passes OPTIMIZER_PASSES]

This is the converter for converting protocol buffer between tf and onnx.

optional arguments:
  -h, --help            show this help message and exit
  --infile INFILE, -i INFILE
                        Input file path, can be pb or ckpt file.
  --outfile OUTFILE, -o OUTFILE
                        Output file path.
  --convert_to {onnx,tf}, -t {onnx,tf}
                        Format converted to.

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
```
