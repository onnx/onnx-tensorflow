from setuptools import setup
import os

TOP_DIR = os.path.realpath(os.path.dirname(__file__))

with open(os.path.join(TOP_DIR, 'VERSION_NUMBER')) as version_file:
    version = version_file.read().strip()

if os.getenv('TRAVIS'):
    # On travis, we install from source, therefore no need to specify version.
    onnx_dep = "onnx"
else:
    # For user, we install the onnx release known to work with our release.
    with open(os.path.join(TOP_DIR, 'ONNX_VERSION_NUMBER')) as onnx_version_file:
        onnx_version = onnx_version_file.read().strip()
        onnx_dep = "onnx>=" + onnx_version

setup(
    name='onnx-tf',
    version=version,
    description='Tensorflow backend and frontend for ONNX (Open Neural Network Exchange).',
    install_requires=[onnx_dep],
    url='https://github.com/onnx/onnx-tensorflow/',
    author='Arpith Jacob, Tian Jin, Gheorghe-Teodor Bercea, Wenhao Hu',
    author_email='tian.jin1@ibm.com',
    license='Apache License 2.0',
    packages=['onnx_tf', 'onnx_tf.backends', 'onnx_tf.frontends'],
    zip_safe=False)
