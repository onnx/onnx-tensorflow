How to implement new op
======

When you get `{} op is not implemented`, you can follow next steps to implement.  
Customize op can also be implemented in similar way.

### Backend

1.  Verify the latest master version of ONNX is installed on your environment
2.  Find specification from [onnx/Operators](https://github.com/onnx/onnx/blob/master/docs/Operators.md).
3.  Implement the handler. All inputs and attrs could get from step 2.
    ```
    - add handler to /onnx_tf/handlers/backend/
    - in the new handler define a classmethod called version_{version}

    * version is the number of since version, which can get from operator's specification
    ```
4.  From within the `onnx_tf` directory, run `gen_opset.py`.
5.  From within the `onnx_tf` directory, run `gen_status.py -m`.
6.  From within the `onnx_tf` directory, run `gen_doc.py` if there is any update to CLI or API.
7.  Verify the operator's test cases in `test/backend/test_onnx_backend.py` all pass.
8.  Add any additional test cases to `test/backend/test_node.py`.
