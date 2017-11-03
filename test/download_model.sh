mkdir -p ../../onnx_models/

wget https://s3.amazonaws.com/download.onnx/models/bvlc_alexnet.tar.gz --directory-prefix=../../onnx_models/
pushd ../../onnx_models/ && tar -xzf bvlc_alexnet.tar.gz && popd

wget https://s3.amazonaws.com/download.onnx/models/densenet121.tar.gz --directory-prefix=../../onnx_models/
pushd ../../onnx_models/ && tar -xzf densenet121.tar.gz && popd

wget https://s3.amazonaws.com/download.onnx/models/inception_v1.tar.gz --directory-prefix=../../onnx_models/
pushd ../../onnx_models/ && tar -xzf inception_v1.tar.gz && popd

wget https://s3.amazonaws.com/download.onnx/models/inception_v2.tar.gz --directory-prefix=../../onnx_models/
pushd ../../onnx_models/ && tar -xzf inception_v2.tar.gz && popd

wget https://s3.amazonaws.com/download.onnx/models/resnet50.tar.gz --directory-prefix=../../onnx_models/
pushd ../../onnx_models/ && tar -xzf resnet50.tar.gz && popd

wget https://s3.amazonaws.com/download.onnx/models/shufflenet.tar.gz --directory-prefix=../../onnx_models/
pushd ../../onnx_models/ && tar -xzf shufflenet.tar.gz && popd

wget https://s3.amazonaws.com/download.onnx/models/squeezenet.tar.gz --directory-prefix=../../onnx_models/
pushd ../../onnx_models/ && tar -xzf squeezenet.tar.gz && popd

wget https://s3.amazonaws.com/download.onnx/models/vgg16.tar.gz --directory-prefix=../../onnx_models/
pushd ../../onnx_models/ && tar -xzf vgg16.tar.gz && popd

wget https://s3.amazonaws.com/download.onnx/models/vgg19.tar.gz --directory-prefix=../../onnx_models/
pushd ../../onnx_models/ && tar -xzf vgg19.tar.gz && popd