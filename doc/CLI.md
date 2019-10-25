ONNX-Tensorflow Command Line Interface
======

## Available commands:
- convert

More information: `onnx-tf -h`
```
usage: onnx-tf [-h] {convert}

ONNX-Tensorflow Command Line Interface

positional arguments:
  {convert}   Available commands.

optional arguments:
  -h, --help  show this help message and exit
```

## Usage:

### Convert:

#### From ONNX to Tensorflow:
`onnx-tf convert -i /path/to/input.onnx -o /path/to/output.pb`

More information: `onnx-tf convert -h`
```
usage: onnx-tf [-h] --infile INFILE --outfile OUTFILE [--device DEVICE]
               [--strict STRICT] [--logging_level LOGGING_LEVEL]

This is the converter for converting protocol buffer between tf and onnx.

optional arguments:
  -h, --help            show this help message and exit
  --infile INFILE, -i INFILE
                        Input file path.
  --outfile OUTFILE, -o OUTFILE
                        Output file path.

backend arguments (onnx -> tf):
  --device DEVICE       The device to execute this model on. (from
                        onnx_tf.backend.prepare)
  --strict STRICT       Whether to enforce semantic equivalence between the
                        original model and the converted tensorflow model,
                        defaults to True (yes, enforce semantic equivalence).
                        Changing to False is strongly discouraged. Currently,
                        the strict flag only affects the behavior of MaxPool
                        and AveragePool ops. (from onnx_tf.backend.prepare)
  --logging_level LOGGING_LEVEL
                        The logging level, default is INFO. Change it to DEBUG
                        to see more conversion details or to WARNING to see
                        less (from onnx_tf.backend.prepare)
```
