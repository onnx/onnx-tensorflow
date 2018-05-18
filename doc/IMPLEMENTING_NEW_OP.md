How to implement new op
======

When you get `{} op is not implemented`, you can follow next steps to implement.  
Customize op can also be implemented in similar way.

### Backend

1.  Find specification from [onnx/Operators](https://github.com/onnx/onnx/blob/master/docs/Operators.md).
2.  Decide if need a specific handler.  
    Op doesn't need a specific handler when tf's specification highly matches onnx's, means:
    
    - inputs are same
    - tf's attributes is a subset of onnx's    
    - attr doesn't exist in tf could be set by `DEFAULT_ONNX_ATTR_PER_OP` in `backend.py`  
    
    otherwise, op needs a specific handler.
3.  Implement. All inputs and attrs could get from step 1.
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
4.  Run `gen_opset.py` and `gen_doc.py`.
5.  Add test case to `test_node.py`.

### Frontend

1.  Find specification from [onnx/Operators](https://github.com/onnx/onnx/blob/master/docs/Operators.md).
2.  A specific handler is needed in frontend in most cases.
3.  Implement.
    ```
    - add handler to frontend_v{version}
    - add decorator register_onnx_op with onnx op name if it is an onnx op
    
    * version is the number of since version, which can get from operator's specification
    ```
4.  Run `gen_opset.py` and `gen_doc.py`.
5.  Add test case to `test_node.py`.