import tensorflow as tf
import numpy as np

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op, ps_description
from onnx_tf.handlers.handler import tf_func


@onnx_op("Unique")
@tf_func(tf.unique_with_counts)
@tf_func(tf.sort)
@ps_description("""
Find the unique elements of a tensor. When an optional attribute 'axis' is provided, unique subtensors sliced along the 'axis' are returned. Otherwise the input tensor is flattened and unique values of the flattened tensor are returned.

This operator returns the unique values or sliced unique subtensors of the input tensor and three optional outputs. The first output tensor 'Y' contains all unique values or subtensors of the input. The second optional output tensor 'indices' contains indices of 'Y' elements' first occurance in 'X'.. The third optional output tensor 'inverse_indices' contains, for elements of 'X', its corresponding indices in 'Y'. ". The fourth optional output tensor 'counts' contains the count of each element of 'Y' in the input.

Outputs are either sorted in ascending order or optionally in the order of the first occurrence of the values in the input.

Attributes:

axis : int
    (Optional) The dimension to apply unique. If not specified, the unique elements of the flattened input are returned. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input).
sorted : int (default is 1)
    (Optional) Whether to sort the unique elements in ascending order before returning as output. Must be one of 0, or 1 (default).

Inputs

X (non-differentiable) : T
    A N-D input tensor that is to be processed.

Outputs (1 - 4)

Y (non-differentiable) : T
    A tensor of the same type as 'X' containing all the unique values or subtensors sliced along a provided 'axis' in 'X', either sorted or maintained in the same order they occur in input 'X'
indices (optional, non-differentiable) : tensor(int64)
    A 1-D INT64 tensor containing indices of 'Y' elements' first occurance in 'X'. When 'axis' is provided, it contains indices to subtensors in input 'X' on the 'axis'. When 'axis' is not provided, it contains indices to values in the flattened input tensor.
inverse_indices (optional, non-differentiable) : tensor(int64)
    A 1-D INT64 tensor containing, for elements of 'X', its corresponding indices in 'Y'. When 'axis' is provided, it contains indices to subtensors in output 'Y' on the 'axis'. When 'axis' is not provided, it contains indices to values in output 'Y'.
counts (optional, non-differentiable) : tensor(int64)
    A 1-D INT64 tensor containing the count of each element of 'Y' in input 'X'

Type Constraints

T : tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
    Input can be of any tensor type.
"""
)
class Unique(BackendHandler):
  @classmethod
  def args_check(cls, node, **kwargs):
    return 0

  @classmethod
  def version_11(cls, node, **kwargs):
    # x, roi, scales and sizes are all in NCHW format
    tensor_dict = kwargs["tensor_dict"]
    X = tensor_dict[node.inputs[0]]
    unique_axis = tensor_dict[node.inputs[1]]
    if_sorted = tensor_dict[node.inputs[2]]

    # sort
    if if_sorted == 1:
        # # method 1, tf api, argsort
        # # tf.argsort returns the indices of a tensor that give its sorted order along an axis.
        # X_indices_sorted = tf.argsort(X, -1, 'ASCENDING')
        # X_sorted = tf.gather(X, X_indices_sorted)
        
        # method 2, tf api, sort
        X_sorted = tf.sort(X, -1, 'ASCENDING')
        
        # # method 3, numpy api
        # X_indices_sorted = np.argsort(X, -1)
        # X_sorted = np.take_along_axis(X, X_indices_sorted, -1)
    else:
        X_sorted = X
        x_indices_aft_sort = None
    # Unique
    # tf.unique_with_counts returns
    # a tensor y containing all of the unique elements of x sorted in the same order that they occur in x.
    # a tensor idx the same size as x that contains the index of each value of x in the unique output y.
    # a third tensor count that contains the count of each element of y in x.
    y, inverse_idx, counts = tf.unique_with_counts(X_sorted)

    indices = []
    for item in y.tolist():
        print(item)
        indices.append(np.argmax(X == item))
    print(indices)

    if if_sorted == 1: 
        inverse_indices = []
        for item in X.tolist():
            inverse_indices.append(np.argmax(y == item))
        print(inverse_indices)
    else:
        inverse_indices = inverse_idx 

    return y, indices, inverse_indices, counts
