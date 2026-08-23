# TensorFlow Backend for ONNX (archived)

> [!IMPORTANT]
> This project is no longer maintained and is being archived.
> The final release is [ONNX-TF 1.10.0](https://github.com/onnx/onnx-tensorflow/releases/tag/v1.10.0), published in March 2022.
> Its final automated tests used ONNX 1.10.2, TensorFlow 2.8.0, and Python 3.8-3.9.
> No compatibility updates, bug fixes, or security fixes will be provided.

ONNX-TF converted ONNX models to TensorFlow and exposed an ONNX backend for
running models through TensorFlow. The source, documentation, releases, and
issue history remain available for historical reference.

## Alternatives

- To run an ONNX model, use a maintained ONNX backend such as
  [ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-python.html).
- For ONNX-to-TensorFlow or ONNX-to-LiteRT/TFLite conversion, the independently
  maintained [onnx2tf](https://github.com/PINTO0309/onnx2tf) project may be
  useful. It is a community project and is not maintained or endorsed by ONNX.
- [tf2onnx](https://github.com/onnx/tensorflow-onnx) converts models in the
  opposite direction, from TensorFlow to ONNX. It is not a replacement for the
  ONNX-to-TensorFlow functionality in this repository.

## Final tested environment

The final automated test configuration used the following versions:

| Component | Version |
| --- | --- |
| ONNX-TF | 1.10.0 |
| ONNX | 1.10.2 |
| TensorFlow | 2.8.0 |
| Python | 3.8-3.9 |

These versions are retained for historical reproducibility. They are old and
may contain unresolved defects or vulnerabilities. Do not assume that this
project works with current Python, ONNX, TensorFlow, Keras, or related
packages.

Historical documentation is available in [`doc/`](doc/), including the
[command-line reference](doc/CLI.md), [Python API](doc/API.md), and
[operator support table](doc/support_status.md). The final Model Zoo results
are preserved in the
[repository wiki](https://github.com/onnx/onnx-tensorflow/wiki/ModelZoo-Status-(branch=master)).

## Support and security

This repository does not accept feature requests, bug fixes, or compatibility
updates. Existing issues and pull requests are retained as historical records.
See [SECURITY.md](SECURITY.md) for the unsupported-version policy.

The project is licensed under Apache License 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).
