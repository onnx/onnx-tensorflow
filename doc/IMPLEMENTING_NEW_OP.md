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
2.  Implement. A specific handler is always needed in frontend.
    Please read doc of `Handler` and `FrontendHandler` first and reference other handler classes.
    ```
    - create a new handler class file under handlers/frontend 
    or add a new version method under op handler class 
    depends on if you want to add a new op or deal with a new version
    - if need, implement args_check method
    - add decorator @onnx_op and @tf_op if add a new op handler class
    
    * version is the number of since version, which can get from operator's specification
    ```
3.  Run `gen_opset.py` and `gen_doc.py`.
4.  Add test case to `test_node.py`.