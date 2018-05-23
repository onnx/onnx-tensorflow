# Tensorflow Backend and Frontend for ONNX
[![Build Status](https://travis-ci.org/onnx/onnx-tensorflow.svg?branch=master)](https://travis-ci.org/onnx/onnx-tensorflow)

[ONNX-Tensorflow API](https://github.com/onnx/onnx-tensorflow/blob/master/doc/API.md)

[ONNX-Tensorflow Op Coverage Status](https://github.com/onnx/onnx-tensorflow/blob/master/doc/support_status.md)

## Tutorials:
[Running an ONNX model using Tensorflow](https://github.com/onnx/tutorials/blob/master/tutorials/OnnxTensorflowImport.ipynb)

[Exporting a Tensorflow Model to ONNX](https://github.com/onnx/tutorials/blob/master/tutorials/OnnxTensorflowExport.ipynb)

## To install:
ONNX-TF requires ONNX (Open Neural Network Exchange) as an external dependency, for any issues related to ONNX installation, we refer our users to [ONNX project repository](https://github.com/onnx/onnx) for documentation and help. Notably, please ensure that protoc is available if you plan to install ONNX via pip.

The specific ONNX release version that we support in the master branch of ONNX-TF can be found [here](https://github.com/onnx/onnx-tensorflow/blob/master/ONNX_VERSION_NUMBER). This information about ONNX version requirement is automatically encoded in `setup.py`, therefore users needn't worry about ONNX version requirement when installing ONNX-TF.

To install the latest version of ONNX-TF via pip, run `pip install onnx-tf`.

Because users often have their own preferences for which variant of Tensorflow to install (i.e., a GPU version instead of a CPU version), we do not explicitly require tensorflow in the installation script. It is therefore users' responsibility to ensure that the proper variant of Tensorflow is available to ONNX-TF. Moreoever, we require Tensorflow version >= 1.5.0.

## To test:
For backend, run `python -m unittest discover test`.

## Example:
In this example, we will define and run a Relu node and print the result.
This example is available as a python script at example/relu.py .
```python
from onnx_tf.backend import run_node
from onnx import helper

node_def = helper.make_node("Relu", ["X"], ["Y"])
output = run_node(node_def, [[-0.1, 0.1]])
print(output["Y"])
```
The result is `[ 0.   0.1]`

## Development Install:
- Install ONNX master branch from source.
- Install Tensorflow>=1.5.0.
- Run `git clone git@github.com:onnx/onnx-tensorflow.git && cd onnx-tensorflow`.
- Run `pip install -e .`.

## Folder Structure:
- __onnx_tf__ main source code file.
- __test__ test files.

## Code Standard:
- Install pylint:
```
pip install pylint
wget -O /tmp/pylintrc https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/ci_build/pylintrc
```
- Check format:
```
pylint --rcfile=/tmp/pylintrc myfile.py
```

## Documentation Standard:
http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

## Test Help:
https://docs.python.org/2/library/unittest.html

## Authors:
Arpith Jacob (IBM Research)

Tian Jin (IBM Research)

Gheorghe-Teodor Bercea (IBM Research)

Wenhao Hu (LeapMind)
