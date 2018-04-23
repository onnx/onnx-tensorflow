# Tensorflow Backend and Frontend for ONNX
[![Build Status](https://travis-ci.org/onnx/onnx-tensorflow.svg?branch=master)](https://travis-ci.org/onnx/onnx-tensorflow)

[ONNX-Tensorflow API](https://github.com/onnx/onnx-tensorflow/blob/master/onnx_tf/doc/API.md)

[ONNX-Tensorflow Op Coverage Status](https://github.com/onnx/onnx-tensorflow/blob/master/onnx_tf/doc/support_status.md)

## Tutorials:
[Running an ONNX model using Tensorflow](https://github.com/onnx/tutorials/blob/master/tutorials/OnnxTensorflowImport.ipynb)

## To install:
Firstly install [ONNX](https://github.com/onnx/onnx) which cannot be installed by pip unless protoc is available.

Secondly install Tensorflow>=1.5.0.

Then, run `pip install onnx-tf`

## To test:
For backend, run `python -m unittest discover test/backend`.

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
- Install ONNX
- Install Tensorflow>=1.5.0
- Run `git clone git@github.com:onnx/onnx-tensorflow.git && cd onnx-tensorflow`
- Run `pip install -e .`
- Development follows conventions [here](https://github.com/onnx/onnx-caffe2/blob/master/onnx_caffe2/backend.py)

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
