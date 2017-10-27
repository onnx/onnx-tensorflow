# Experimental Tensorflow Backend for ONNX

## To install:
run `pip install onnx-tf`

## To test:
run `python -m unittest discover test`

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
- Git clone
- Run `pip install -e .` on the root directory.
- Backend dev follows conventions [here](https://github.com/onnx/onnx-caffe2/blob/master/onnx_caffe2/backend.py).
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
Arpith Jacob

Tian Jin

Gheorghe-Teodor Bercea
