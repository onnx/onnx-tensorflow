#!/bin/bash

scripts_dir=$(dirname $(readlink -e "${BASH_SOURCE[0]}"))
source "$scripts_dir/common";

pip install tensorflow

onnx_tf_dir="$PWD"
pip install -e $onnx_tf_dir

python --version

# Make sure we run through all tests.
set +e

python -m unittest discover test/backend/
python -m unittest discover test/frontend/

# Back to exit on error.
set -e