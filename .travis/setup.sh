export TRAVIS=1

echo "This is a CI test for onnx-tf master with latest onnx and tf."
docker build -t=test-image ./.travis/onnx-master_tf-nightly
docker run -t -d --name=test test-image /bin/bash
