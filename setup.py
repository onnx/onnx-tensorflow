from setuptools import setup

setup(
    name='onnx-tf',
    version='1.1',
    description='Tensorflow backend and frontend for ONNX (Open Neural Network Exchange).',
    install_requires=['onnx'],
    url='https://github.com/tjingrant',
    author='Arpith Jacob, Tian Jin, Gheorghe-Teodor Bercea',
    author_email='tian.jin1@ibm.com',
    license='Apache License 2.0',
    packages=['onnx_tf', 'onnx_tf.backends', 'onnx_tf.frontends'],
    zip_safe=False)
