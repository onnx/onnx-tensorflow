export TRAVIS=1

echo "This is a CI test for onnx-tf master with latest onnx and tf final 1.x release."
docker build -t=test-image ./.travis/onnx-master_tf-1.x-release
docker run -t -d --name=test test-image /bin/bash
