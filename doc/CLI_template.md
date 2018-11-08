ONNX-Tensorflow Command Line Interface
======

## Available commands:
- convert

More information: `onnx-tf -h`
```
{onnx-tf -h}
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
{onnx-tf convert -h}
```
