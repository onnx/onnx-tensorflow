# Tensorflow Backend for ONNX
[![Build Status](https://travis-ci.org/onnx/onnx-tensorflow.svg?branch=master)](https://travis-ci.org/onnx/onnx-tensorflow)

## To convert models from ONNX to Tensorflow:

### Use CLI:

[Command Line Interface Documentation](https://github.com/onnx/onnx-tensorflow/blob/master/doc/CLI.md)

From ONNX to Tensorflow: `onnx-tf convert -i /path/to/input.onnx -o /path/to/output.pb`

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

To install the latest version of ONNX-TF via pip, run `pip install onnx-tf`.

Because users often have their own preferences for which variant of Tensorflow to install (i.e., a GPU version instead of a CPU version), we do not explicitly require tensorflow in the installation script. It is therefore users' responsibility to ensure that the proper variant of Tensorflow is available to ONNX-TF. Moreover, we require Tensorflow version == 2.2.0.

## Development:

### Coverage Status:
[ONNX-Tensorflow Op Coverage Status](https://github.com/onnx/onnx-tensorflow/blob/master/doc/support_status.md)

### API:
[ONNX-Tensorflow API](https://github.com/onnx/onnx-tensorflow/blob/master/doc/API.md)

### Installation:
- Run `git clone https://github.com/onnx/onnx.git && cd onnx`.
- Run `git submodule update --init --recursive`.
- Run `pip install -e .`.
- Install Tensorflow >= 2.2.0 and tensorflow-addons. (Note for Tensorflow 1.x please refer the [tf-1.x branch](https://github.com/onnx/onnx-tensorflow/tree/tf-1.x))
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

PS. Please ensure your code is backward compatible with older version of ONNX. You can easily test it by running the following [docker container](https://hub.docker.com/r/winnietsang/onnx-tensorflow) with your code. If you don't have Docker installed yet, please follow this link to install [Docker](https://docs.docker.com/install/) on your environment.
```
sudo docker pull winnietsang/onnx-tensorflow:onnx1.7.0-tf2.2
sudo docker run -it --name=YOUR-CONTAINER-NAME winnietsang/onnx-tensorflow:onnx1.7.0-tf2.2 /bin/bash
git clone https://github.com/YOUR-USERNAME/onnx-tensorflow.git
cd onnx-tensorflow
git checkout -b YOUR-BRANCH --track remotes/origin/YOUR-BRANCH
pip3 install -e .
python3 -m unittest discover test
```

#### Test Help:
https://docs.python.org/2/library/unittest.html
