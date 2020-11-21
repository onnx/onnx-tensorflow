# Tensorflow Backend for ONNX
[![Build Status](https://travis-ci.com/onnx/onnx-tensorflow.svg?branch=master)](https://travis-ci.com/github/onnx/onnx-tensorflow)

## To convert models from ONNX to Tensorflow:

### Use CLI:

[Command Line Interface Documentation](https://github.com/onnx/onnx-tensorflow/blob/master/doc/CLI.md)

From ONNX to Tensorflow: `onnx-tf convert -i /path/to/input.onnx -o /path/to/output`

### Convert programmatically:

[From ONNX to Tensorflow](https://github.com/onnx/onnx-tensorflow/blob/master/example/onnx_to_tf.py)

### Migrating from `onnx-tf` to `tf-onnx`:
We have joined force with Microsoft to co-develop ONNX Tensorflow frontend.
For current onnx-tf frontend users, please migrate to use tf-onnx (https://github.com/onnx/tensorflow-onnx) where our code had been merged into.

## ONNX model inference with Tensorflow backend:
```
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("input_path")  # load onnx model
output = prepare(onnx_model).run(input)  # run the loaded model
```

## More tutorials:
[Running an ONNX model using Tensorflow](https://github.com/onnx/tutorials/blob/master/tutorials/OnnxTensorflowImport.ipynb)

## Production Installation:
ONNX-TF requires ONNX (Open Neural Network Exchange) as an external dependency, for any issues related to ONNX installation, we refer our users to [ONNX project repository](https://github.com/onnx/onnx) for documentation and help. Notably, please ensure that protoc is available if you plan to install ONNX via pip.

The specific ONNX release version that we support in the master branch of ONNX-TF can be found [here](https://github.com/onnx/onnx-tensorflow/blob/master/ONNX_VERSION_NUMBER). This information about ONNX version requirement is automatically encoded in `setup.py`, therefore users needn't worry about ONNX version requirement when installing ONNX-TF.

As of November 24, 2020, we are unable to publish release 1.7.0 to PyPi due to problem described in issue #738.\
Once the issue is resolved you should install the latest version of ONNX-TF via pip, by running `pip install onnx-tf`\
In the mean time please get release 1.7.0 by running the following commands to checkout v1.7.0 tag and install it from source via pip.\
`git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow`\
`git checkout v1.7.0`\
`pip install -e .`

Because users often have their own preferences for which variant of Tensorflow to install (i.e., a GPU version instead of a CPU version), we do not explicitly require tensorflow in the installation script. It is therefore users' responsibility to ensure that the proper variant of Tensorflow is available to ONNX-TF. Moreover, we require Tensorflow version == 2.3.1.

## Development:

### Coverage Status:
[ONNX-Tensorflow Op Coverage Status](https://github.com/onnx/onnx-tensorflow/blob/master/doc/support_status.md)

### API:
[ONNX-Tensorflow API](https://github.com/onnx/onnx-tensorflow/blob/master/doc/API.md)

### Installation:
- Install ONNX master branch from source. 
- Install Tensorflow >= 2.3.1 and tensorflow-addons. (Note for Tensorflow 1.x please refer the [tf-1.x branch](https://github.com/onnx/onnx-tensorflow/tree/tf-1.x))
- Run `git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow`.
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
