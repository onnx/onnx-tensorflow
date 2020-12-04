# Tensorflow Backend for ONNX

## To convert models from ONNX to Tensorflow:

### Use CLI:

[Command Line Interface Documentation](https://github.com/onnx/onnx-tensorflow/blob/tf-1.x/doc/CLI.md)

From ONNX to Tensorflow: `onnx-tf convert -i /path/to/input.onnx -o /path/to/output.pb`

### Convert programmatically:

[From ONNX to Tensorflow](https://github.com/onnx/onnx-tensorflow/blob/tf-1.x/example/onnx_to_tf.py)

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

Because users often have their own preferences for which variant of Tensorflow to install (i.e., a GPU version instead of a CPU version), we do not explicitly require tensorflow in the installation script. It is therefore users' responsibility to ensure that the proper variant of Tensorflow is available to ONNX-TF. Moreover, we require Tensorflow version == 1.15.4.

To install the latest version of ONNX-TF v1.7.0
- Run `git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow`.
- Run `git checkout v1.7.0-tf-1.15`.
- Run `pip install -e .`.

## Development:

### Coverage Status:
[ONNX-Tensorflow Op Coverage Status](https://github.com/onnx/onnx-tensorflow/blob/tf-1.x/doc/support_status.md)

### API:
[ONNX-Tensorflow API](https://github.com/onnx/onnx-tensorflow/blob/tf-1.x/doc/API.md)

### Installation:
- Install ONNX master branch from source.
- Install Tensorflow 1.15.4. (For Tensorflow 2.x support please refer [here](https://github.com/onnx/onnx-tensorflow/blob/master/README.md/).)
- Run `git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow`.
- Run `git checkout tf-1.x`.
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

#### Note:
Branch tf-1.x is for users who cannot upgrade to Tensorflow 2.x yet. This branch will only support up to ONNX OpSet 12 operators. If any user needs to use operators in OpSet 13 or above, please upgrade Tensoflow to 2.x and use the master branch of this repo. By January 1st, 2021 this branch will switch to maintenance mode only, no new development will be added into this branch from then on.
