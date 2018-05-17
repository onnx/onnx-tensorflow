#!/bin/bash

scripts_dir=$(dirname $(readlink -e "${BASH_SOURCE[0]}"))
source "$scripts_dir/common"

# Install protobuf.
pb_dir="$build_cache_dir/pb"
mkdir -p $pb_dir
wget -qO- "https://github.com/google/protobuf/releases/download/v$PB_VERSION/protobuf-$PB_VERSION.tar.gz" | tar -xvz -C "$pb_dir" --strip-components 1
ccache -z
cd "$pb_dir" && ./configure && make && make check && sudo make install && sudo ldconfig
ccache -s

# Install onnx.
onnx_dir="$workdir/onnx"
mkdir -p $onnx_dir
cd "$onnx_dir" && git clone --recursive git://github.com/onnx/onnx.git

# Checkout the version specified.
cd onnx
if [ "$ONNX_BUILD_FROM" = "MASTER" ]
then
  git checkout master
else
  ONNX_BRANCH=rel-$(cat $TRAVIS_BUILD_DIR/ONNX_VERSION_NUMBER)
  echo "Using ONNX branch:"
  echo $ONNX_BRANCH
  git checkout $ONNX_BRANCH
fi
# Install onnx.
pip install -e .