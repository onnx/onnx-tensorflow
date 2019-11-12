How to implement new op
======

When you get `{} op is not implemented`, you can follow next steps to implement.  
Customize op can also be implemented in similar way.

### Backend

1.  Verify the latest master version of ONNX is installed on your environment
2.  Find specification from [onnx/Operators](https://github.com/onnx/onnx/blob/master/docs/Operators.md).
3.  Decide if need a specific handler.
    Op doesn't need a specific handler when tf's specification highly matches onnx's, means:
    
    - inputs are same
    - tf's attributes is a subset of onnx's    
    - attr doesn't exist in tf could be set by `DEFAULT_ONNX_ATTR_PER_OP` in `backend.py`  
    
    otherwise, op needs a specific handler.
4.  Implement. All inputs and attrs could get from step 2.
    ```
    non-specific handler
    
    - update ONNX_OP_TO_TF_OP in common.py
    - if need, update DEFAULT_ONNX_ATTR_PER_OP
    ```
    ```
    specific handler
    
    - add handler to backend_v{version}

    * version is the number of since version, which can get from operator's specification
    ```
5.  Run `gen_opset.py`.
6.  Run `gen_status.py -v master`.
7.  Run `gen_doc.py` if there is any update to CLI or API.
8.  Add test case to `test_node.py`.
