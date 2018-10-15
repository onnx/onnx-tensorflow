# Tensorflow Backend and Frontend for ONNX
[![Build Status](https://travis-ci.org/onnx/onnx-tensorflow.svg?branch=master)](https://travis-ci.org/onnx/onnx-tensorflow)

## To convert models between Tensorflow and ONNX:

### Use CLI:

[Command Line Interface Documentation](https://github.com/onnx/onnx-tensorflow/blob/master/doc/CLI.md)

From Tensorflow to ONNX: `onnx-tf convert -t onnx -i /path/to/input.pb -o /path/to/output.onnx --ignore_unimplemented True`

From ONNX to Tensorflow: `onnx-tf convert -t tf -i /path/to/input.onnx -o /path/to/output.pb`

### Convert programmatically:

[From Tensorflow to ONNX](https://github.com/onnx/onnx-tensorflow/blob/master/example/tf_to_onnx.py)

[From ONNX to Tensorflow](https://github.com/onnx/onnx-tensorflow/blob/master/example/onnx_to_tf.py)

## ONNX model inference with Tensorflow backend:
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
