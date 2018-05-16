#!/bin/bash

scripts_dir=$(dirname $(readlink -e "${BASH_SOURCE[0]}"))
source "$scripts_dir/common"

# install protobuf.
pb_dir="$build_cache_dir/pb"
mkdir -p $pb_dir

wget -qO- "https://github.com/google/protobuf/releases/download/v$PB_VERSION/protobuf-all-$PB_VERSION.tar.gz" | tar -xvz -C "$pb_dir" --strip-components 1
ccache -z
cd "$pb_dir" && ./configure && make && make check && sudo make install && sudo ldconfig
cd "$pb_dir/python" && python setup.py install
ccache -s

# install onnx.
onnx_dir="$workdir/onnx"
mkdir -p $onnx_dir
cd "$onnx_dir" && git clone --recursive git://github.com/onnx/onnx.git
# checkout the version specified.
cd onnx && git checkout $ONNX_BRANCH
# install onnx.
pip install -e .