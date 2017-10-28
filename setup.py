from setuptools import setup

setup(name='onnx-tf',
      version='0.1.1',
      description='Tensorflow backend for ONNX.',
      install_requires=['onnx==0.2', 'tensorflow', 'numpy'],
      url='https://github.com/tjingrant',
      author='Arpith Jacob, Tian Jin, Gheorghe-Teodor Bercea',
      author_email='tian.jin1@ibm.com',
      license='Apache License 2.0',
      packages=['onnx_tf'],
      zip_safe=False)
