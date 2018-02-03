from setuptools import setup

setup(name='onnx-tf',
      version='1.0',
      description='Tensorflow backend for ONNX (Open Neural Network Exchange).',
      # as per https://github.com/tensorflow/tensorflow/issues/16488
      # need to bump numpy version manually.
      install_requires=['onnx', 'tensorflow', 'numpy>=1.14'],
      url='https://github.com/tjingrant',
      author='Arpith Jacob, Tian Jin, Gheorghe-Teodor Bercea',
      author_email='tian.jin1@ibm.com',
      license='Apache License 2.0',
      packages=['onnx_tf'],
      zip_safe=False)
