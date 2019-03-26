export TRAVIS=1

# Check if head has a tag?
git describe --exact-match --tags HEAD
if [ $? -eq 0 ]; then
	echo "This is a release test for onnx-tf."
	export DOCKER_CONTAINER_NAME="$(git describe --exact-match --tags HEAD)"
	echo "Docker container is determined to be ${DOCKER_CONTAINER_NAME}."
	docker pull winnietsang/onnx-tensoflow:${DOCKER_CONTAINER_NAME}
	docker run -t -d --name=test winnietsang/onnx-tensoflow:${DOCKER_CONTAINER_NAME} /bin/bash
else
	echo "This is a non-release test for onnx-tf."
	docker build -t=test-image ./.travis/onnx-master_tf-nightly
	docker run -t -d --name=test test-image /bin/bash
fi
