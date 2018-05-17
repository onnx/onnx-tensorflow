#!/bin/bash

scripts_dir=$(dirname $(readlink -e "${BASH_SOURCE[0]}"))
source "$scripts_dir/common";

pip install tensorflow

onnx_tf_dir="$PWD"
pip install -e $onnx_tf_dir

python --version

python -m unittest discover test