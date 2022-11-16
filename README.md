# TensorFlow Backend for ONNX
![Backend Test Status](https://github.com/onnx/onnx-tensorflow/workflows/Backend%20test/badge.svg)
![ModelZoo Test Status](https://github.com/onnx/onnx-tensorflow/workflows/ModelZoo%20test/badge.svg)

### Note this repo is not actively maintained and will be deprecated. If you are interested in becoming the owner, please contact the ONNX Steering Committee (https://github.com/onnx/steering-committee).

[Open Neural Network Exchange (ONNX)](https://onnx.ai) is an open standard format for representing machine learning models. ONNX is supported by a community of partners who have implemented it in many frameworks and tools.

TensorFlow Backend for ONNX makes it possible to use ONNX models as input for [TensorFlow](https://www.tensorflow.org). The ONNX model is first converted to a TensorFlow model and then delegated for execution on TensorFlow to produce the output.

This is one of the two TensorFlow converter projects which serve different purposes in the ONNX community:
- [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow) converts ONNX models to Tensorflow
- [tf2onnx](https://github.com/onnx/tensorflow-onnx) converts Tensorflow models to ONNX

## Converting Models from ONNX to TensorFlow

### Use CLI

[Command Line Interface Documentation](https://github.com/onnx/onnx-tensorflow/blob/master/doc/CLI.md)

From ONNX to TensorFlow: `onnx-tf convert -i /path/to/input.onnx -o /path/to/output`

### Convert Programmatically

[From ONNX to TensorFlow](https://github.com/onnx/onnx-tensorflow/blob/master/example/onnx_to_tf.py)

### Migrating from `onnx-tf` to `tf-onnx`
We have joined force with Microsoft to co-develop ONNX TensorFlow frontend.
For current onnx-tf frontend users, please migrate to use tf-onnx (https://github.com/onnx/tensorflow-onnx) where our code had been merged into.

## ONNX Model Inference with TensorFlow Backend
```
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("input_path")  # load onnx model
output = prepare(onnx_model).run(input)  # run the loaded model
```

## More Tutorials
[Running an ONNX model using TensorFlow](https://github.com/onnx/tutorials/blob/master/tutorials/OnnxTensorflowImport.ipynb)

## Production Installation
ONNX-TF requires ONNX (Open Neural Network Exchange) as an external dependency, for any issues related to ONNX installation, we refer our users to [ONNX project repository](https://github.com/onnx/onnx) for documentation and help. Notably, please ensure that `protoc` is available if you plan to install ONNX via pip.

The specific ONNX release version that we support in the master branch of ONNX-TF can be found [here](https://github.com/onnx/onnx-tensorflow/blob/master/ONNX_VERSION_NUMBER). This information about ONNX version requirement is automatically encoded in `setup.py`, therefore users needn't worry about ONNX version requirement when installing ONNX-TF.

To install the latest version of ONNX-TF via pip, run `pip install onnx-tf`.

Because users often have their own preferences for which variant of TensorFlow to install (i.e., a GPU version instead of a CPU version), we do not explicitly require tensorflow in the installation script. It is therefore users' responsibility to ensure that the proper variant of TensorFlow is available to ONNX-TF. Moreover, we require TensorFlow version == 2.8.0.

## Development

### Coverage Status
[ONNX-TensorFlow Op Coverage Status](https://github.com/onnx/onnx-tensorflow/blob/master/doc/support_status.md)

### API
[ONNX-TensorFlow API](https://github.com/onnx/onnx-tensorflow/blob/master/doc/API.md)

### Installation
- Install ONNX master branch from source.
- Install TensorFlow >= 2.8.0, tensorflow-probability and tensorflow-addons. (Note TensorFlow 1.x is no longer supported)
- Run `git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow`.
- Run `pip install -e .`.

### Folder Structure
- __onnx_tf__: main source code file.
- __test__: test files.

### Code Standard
- Format code
```
pip install yapf
yapf -rip --style="{based_on_style: google, indent_width: 2}" $FilePath$
```
- Install pylint
```
pip install pylint
wget -O /tmp/pylintrc https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/ci_build/pylintrc
```
- Check format
```
pylint --rcfile=/tmp/pylintrc myfile.py
```

### Documentation Standard
[Google Style Python Docstrings](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)

## Testing

### Unit Tests

To perfom [unit tests](https://docs.python.org/3/library/unittest.html):

```
pip install pytest tabulate
python -m unittest discover test
```

Note: Only the ONNX backend tests found in [`test_onnx_backend.py`](https://github.com/onnx/onnx-tensorflow/blob/master/test/backend/test_onnx_backend.py) require the `pytest` and `tabulate` packages.

Testing requires significant hardware resources, but nonetheless, we highly recommend that users run through the complete test suite before deploying onnx-tf. The complete test suite typically takes between 15 and 45 minutes to complete, depending on hardware configurations.

### Model Zoo Tests

The tests in [`test_modelzoo.py`](https://github.com/onnx/onnx-tensorflow/blob/master/test/test_modelzoo.py) verify whether the [ONNX Model Zoo](https://github.com/onnx/models) models can be successfully validated against the ONNX specification and converted to a TensorFlow representation. Model inferencing on the converted model is not tested currently.

#### Prerequisites

The model zoo uses [Git LFS (Large File Storage)](https://git-lfs.github.com/) to store ONNX model files. Make sure that Git LFS is installed on your operating system.

#### Running

By default, the tests assume that the model zoo repository has been cloned into this project directory. The model zoo directory is scanned for ONNX models. For each model found: download the model, convert the model to TensorFlow, generate a test status, and delete the model. By default, the generated test report is created in the system temporary directory. Run `python test/test_modelzoo.py -h` for help on command line options.

```
git clone https://github.com/onnx/models
python test/test_modelzoo.py
```

Testing all models can take at least an hour to complete, depending on hardware configuration and model download times. If you expect to test some models frequently, we recommend using Git LFS to download those models before running the tests so the large files are cached locally.

#### Reports

When making code contributions, the model zoo tests are run when a commit is merged. Generated test reports are published on the [onnx-tensorflow wiki](https://github.com/onnx/onnx-tensorflow/wiki/ModelZoo-Status-(branch=master)).
