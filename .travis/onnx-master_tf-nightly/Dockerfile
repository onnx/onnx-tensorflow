# Using this script to build your ONNX image.
# Execute "docker build dir", where dir contains this Dockerfile. Make sure you give docker container at least 8GB memory.
# Execute "docker run -i -t image_id", where image_id is the id of the image you just generated.
# Try "python tutorial-without-mobile-part.py".

FROM ubuntu:18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         ssh \
         vim \
         curl \
         ca-certificates \
         wget \
         unzip \
         libjpeg-dev \
         libpng-dev \
         libgflags-dev \
         libgoogle-glog-dev \
         liblmdb-dev \
         libprotobuf-dev \
         protobuf-compiler \
         cmake \
         liblapack3 \
         liblapack-dev \
         libopenblas-base \
         libopenblas-dev \
         liblapacke-dev \
         liblapack-dev \
         python python-pip python-dev python-setuptools \
         python3 python3-pip python3-dev python3-setuptools && \
     rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN apt remove -y python-pip
RUN pip install wheel ipython==5.0 numpy scipy pyyaml pytest

RUN pip3 install --upgrade pip
RUN apt remove -y python3-pip
RUN pip3 install wheel ipython==5.0 numpy scipy pyyaml pytest

RUN mkdir -p /root/programs

# Install ONNX
RUN git clone --recursive https://github.com/onnx/onnx.git /root/programs/onnx
RUN cd /root/programs/onnx; pip2 install -e .
RUN cd /root/programs/onnx; pip3 install -e .

# Install Tensorflow
RUN pip2 install -U tf-nightly
RUN pip3 install -U tf-nightly

WORKDIR /root/programs
