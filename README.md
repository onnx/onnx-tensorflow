# Tensorflow Backend and Frontend for ONNX
[![Build Status](https://travis-ci.org/onnx/onnx-tensorflow.svg?branch=master)](https://travis-ci.org/onnx/onnx-tensorflow)

## To convert pb between Tensorflow and ONNX:

### Use CLI:
Tensorflow -> ONNX: `onnx-tf convert -t onnx -i /path/to/input.pb -o /path/to/output.onnx`

ONNX -> Tensorflow: `onnx-tf convert -t tf -i /path/to/input.onnx -o /path/to/output.pb`

### Use python:

Tensorflow -> ONNX:
```
from tensorflow.core.framework import graph_pb2

from onnx_tf.frontend import tensorflow_graph_to_onnx_model


graph_def = graph_pb2.GraphDef()
with open(input_path, "rb") as f:
  graph_def.ParseFromString(f.read())
nodes, node_inputs = set(), set()
for node in graph_def.node:
  nodes.add(node.name)
  node_inputs.update(set(node.input))
  output = list(set(nodes) - node_inputs)

model = tensorflow_graph_to_onnx_model(graph_def, output, ignore_unimplemented=True)
with open(output_path, 'wb') as f:
  f.write(model.SerializeToString())
```

ONNX -> Tensorflow:
```
import onnx

from onnx_tf.backend import prepare


onnx_model = onnx.load(input_path)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(output_path)
```

## To do inference on ONNX model by using Tensorflow backend:
```
import onnx

from onnx_tf.backend import prepare


output = prepare(onnx.load(input_path)).run(input)
```

## More tutorials:
[Running an ONNX model using Tensorflow](https://github.com/onnx/tutorials/blob/master/tutorials/OnnxTensorflowImport.ipynb)

[Exporting a Tensorflow Model to ONNX](https://github.com/onnx/tutorials/blob/master/tutorials/OnnxTensorflowExport.ipynb)

## Production Installation:
ONNX-TF requires ONNX (Open Neural Network Exchange) as an external dependency, for any issues related to ONNX installation, we refer our users to [ONNX project repository](https://github.com/onnx/onnx) for documentation and help. Notably, please ensure that protoc is available if you plan to install ONNX via pip.

The specific ONNX release version that we support in the master branch of ONNX-TF can be found [here](https://github.com/onnx/onnx-tensorflow/blob/master/ONNX_VERSION_NUMBER). This information about ONNX version requirement is automatically encoded in `setup.py`, therefore users needn't worry about ONNX version requirement when installing ONNX-TF.

To install the latest version of ONNX-TF via pip, run `pip install onnx-tf`.

Because users often have their own preferences for which variant of Tensorflow to install (i.e., a GPU version instead of a CPU version), we do not explicitly require tensorflow in the installation script. It is therefore users' responsibility to ensure that the proper variant of Tensorflow is available to ONNX-TF. Moreoever, we require Tensorflow version >= 1.5.0.

## Development:

### Coverage Status:
[ONNX-Tensorflow Op Coverage Status](https://github.com/onnx/onnx-tensorflow/blob/master/doc/support_status.md)

### API:
[ONNX-Tensorflow API](https://github.com/onnx/onnx-tensorflow/blob/master/doc/API.md)

### Installation:
- Install ONNX master branch from source.
- Install Tensorflow>=1.5.0.
- Run `git clone git@github.com:onnx/onnx-tensorflow.git && cd onnx-tensorflow`.
- Run `pip install -e .`.

### Folder Structure:
- __onnx_tf__ main source code file.
- __test__ test files.

### Code Standard:
- Format code:
```
pip install yapf
yapf -rip --style="{based_on_style: google, indent_width: 2}" $FilePath$
```
- Install pylint:
```
pip install pylint
wget -O /tmp/pylintrc https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/ci_build/pylintrc
```
- Check format:
```
pylint --rcfile=/tmp/pylintrc myfile.py
```

### Documentation Standard:
http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

### To test:
To perfom unit tests, run `python -m unittest discover test`.
Testing requires significant hardware resources, but nonetheless, we highly recommend that users run through the complete test suite before deploying onnx-tf. The complete test suite typically takes between 15 and 45 minutes to complete, depending on hardware configurations.

#### Test Help:
https://docs.python.org/2/library/unittest.html

## Authors:
Arpith Jacob (IBM Research)

Tian Jin (IBM Research)

Gheorghe-Teodor Bercea (IBM Research)

Wenhao Hu (LeapMind)
