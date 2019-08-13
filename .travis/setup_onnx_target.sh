export TRAVIS=1

echo "This is a CI test for onnx-tf master with onnx 1.5.0 and tf stable release."
docker pull winnietsang/onnx-tensorflow:onnx1.5.0-tf1.14.0
docker run -t -d --name=test winnietsang/onnx-tensorflow:onnx1.5.0-tf1.14.0 /bin/bash
