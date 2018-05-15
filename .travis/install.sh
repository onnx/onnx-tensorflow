#!/bin/bash

scripts_dir=$(dirname $(readlink -e "${BASH_SOURCE[0]}"))
source "$scripts_dir/common"

# install onnx.
onnx_dir="$workdir/onnx"
mkdir -p $onnx_dir
cd "$onnx_dir" && git clone --recursive git://github.com/onnx/onnx.git
# checkout the version specified.
cd onnx && git checkout $ONNX_BRANCH
# install onnx.
pip install -e .