export TRAVIS=1

# Check if head has a tag?
git describe --exact-match --tags HEAD
if [ $? -eq 0 ]; then
	echo "This is a release test for onnx-tf."
	export DOCKER_CONTAINER_NAME="$(git describe --exact-match --tags HEAD)"
else
	echo "This is a non-release test for onnx-tf."
	# TODO: switch to a docker file that always pulls the tip of onnx and tf.
	export DOCKER_CONTAINER_NAME="onnx1.4.1-tf1.13.1"
fi

echo "Docker container is determined to be ${DOCKER_CONTAINER_NAME}."

docker pull winnietsang/onnx-tensoflow:${DOCKER_CONTAINER_NAME}