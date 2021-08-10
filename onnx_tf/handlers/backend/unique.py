import tensorflow as tf
import numpy as np

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op, ps_description
from onnx_tf.handlers.handler import tf_func

from onnx_tf.common import exception
from onnx_tf.common import data_type
from onnx_tf.common import sys_config

tf.config.run_functions_eagerly(True)


@onnx_op("Unique")
@tf_func(tf.unique_with_counts)
@tf_func(tf.sort)
@ps_description("""
Find the unique elements of a tensor. 
When an optional attribute 'axis' is provided, unique subtensors sliced along the 'axis' are returned. 
Otherwise the input tensor is flattened and unique values of the flattened tensor are returned.
This operator returns the unique values or sliced unique subtensors of the input tensor and three optional outputs. 
The first output tensor 'Y' contains all unique values or subtensors of the input. 
The second optional output tensor 'indices' contains indices of 'Y' elements' first occurance in 'X'.. 
The third optional output tensor 'inverse_indices' contains, for elements of 'X', its corresponding indices in 'Y'. 
The fourth optional output tensor 'counts' contains the count of each element of 'Y' in the input.
Outputs are either sorted in ascending order or in the order of the first occurrence of the values in the input.
Attributes:
axis : int
    (Optional) The dimension to apply unique. If not specified, the unique elements of the flattened input are returned. 
    Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input).
sorted : int (default is 1)
    (Optional) Whether to sort the unique elements in ascending order before returning as output. 
    Must be one of 0, or 1 (default).
Inputs:
X (non-differentiable) : T
    A N-D input tensor that is to be processed.
Outputs (1 - 4):
Y (non-differentiable) : T
    A tensor of the same type as 'X' containing all the unique values or subtensors sliced along a provided 'axis' in 'X',
    either sorted or maintained in the same order they occur in input 'X'
indices (optional, non-differentiable) : tensor(int64)
    A 1-D INT64 tensor containing indices of 'Y' elements' first occurance in 'X'. When 'axis' is provided, 
    it contains indices to subtensors in input 'X' on the 'axis'. When 'axis' is not provided, 
    it contains indices to values in the flattened input tensor.
inverse_indices (optional, non-differentiable) : tensor(int64)
    A 1-D INT64 tensor containing, for elements of 'X', its corresponding indices in 'Y'. 
    When 'axis' is provided, it contains indices to subtensors in output 'Y' on the 'axis'. When 'axis' is not provided, 
    it contains indices to values in output 'Y'.
counts (optional, non-differentiable) : tensor(int64)
    A 1-D INT64 tensor containing the count of each element of 'Y' in input 'X'

Type Constraints

T : tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), 
    tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), 
    tensor(complex128)
Input can be of any tensor type.
"""
                )
class Unique(BackendHandler):
    x_supported_types = [
        tf.uint8, tf.uint16, tf.uint32, tf.uint64,
        tf.int8, tf.int16, tf.int32, tf.int64,
        tf.float16, tf.float, tf.double, tf.string, tf.bool,
        tf.complex64, tf.complex128
    ]
    axis_supported_types = [
        int
    ]
    sorted_supported_type = axis_supported_types
    x_cast_map = {tf.uint32: tf.int64, tf.bool: None, tf.string: None}
    axis_case_map = {tf.uint8: tf.int, tf.uint16: tf.int}
    sorted_case_map = axis_case_map
    @classmethod
    def args_check(cls, node, **kwargs):
        cls.x_cast_map[tf.uint8] = tf.int8 if sys_config.auto_cast else None
        cls.x_cast_map[tf.uint16] = tf.float16 if sys_config.auto_cast else None
        cls.x_cast_map[tf.uint32] = tf.float32 if sys_config.auto_cast else None
        cls.x_cast_map[tf.uint64] = tf.float64 if sys_config.auto_cast else None
        cls.x_cast_map[tf.complex64] = tf.float64 if sys_config.auto_cast else None
        cls.x_cast_map[tf.complex128] = tf.float128 if sys_config.auto_cast else None
        cls.x_cast_map[tf.uint64] = tf.float64 if sys_config.auto_cast else None

        # Input:
        # data type is acceptable
        x = kwargs["tensor_dict"][node.inputs[0]]
        # x_shape = x.get_shape().as_list()
        # if len(x_shape) <= 0:  # meaning less
        #     exception.OP_UNSUPPORTED_EXCEPT("Unique required N-D input", "Tensorflow")
        x_dtype = x.dtype
        if x_dtype in cls.x_cast_map and cls.x_cast_map[x_dtype] is None:  # why is and
            exception.DTYPE_NOT_CAST_EXCEPT(
                "Unique input " + node.inputs[0] + " with data type '" +
                data_type.tf_to_np_str(x_dtype) + "'",
                data_type.tf_to_np_str_list(cls.x_supported_types))
        # Attributes:
        # axis: Optional, int, range is [-r, r - 1] where r = rank(input), default None.
        unique_axis = node.attrs.get("axis", -1)
        axis_type = unique_axis.dtype
        if not (axis_type in cls.axis_supported_types or axis_type is None):
            exception.DTYPE_NOT_CAST_EXCEPT(
                "Unique axis " + unique_axis + " with data type '" +
                data_type.tf_to_np_str(axis_type) + "'",
                data_type.tf_to_np_str_list(cls.axis_supported_types)
            )
        rank_x = tf.rank(x).numpy()
        if unique_axis >= rank_x or unique_axis < (0 - rank_x):
            exception.OP_UNSUPPORTED_EXCEPT("Unique required axis: None or in rand [-r, r - 1] where r = rank(input).")
        # Attributes:
        # sorted: Optional, int, (0 or 1), default is 1.
        if_sorted = node.attrs.get("sorted", 1)
        if_sorted_type = if_sorted.dtype
        if not (if_sorted_type in cls.sorted_supported_types or if_sorted_type is None):
            exception.DTYPE_NOT_CAST_EXCEPT(
                "Unique sort " + if_sorted + " with data type '" +
                data_type.tf_to_np_str(if_sorted_type) + "'",
                data_type.tf_to_np_str_list(cls.sorted_supported_types)
            )
        if if_sorted != 0 and if_sorted != 1:
            exception.OP_UNSUPPORTED_EXCEPT("Unique required sort: None, either 0 or 1.")

        return 0

    @classmethod
    def version_11(cls, node, **kwargs):
        # get attributions
        tensor_dict = kwargs["tensor_dict"]
        X = tensor_dict[node.inputs[0]]
        print("\n\nX: {1}".format(X))
        unique_axis = node.attrs.get("axis", -1)
        if_sorted = node.attrs.get("sorted", 1)
        print("if_sorted : {1}".format(if_sorted))

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
        print("shape of X after sorted: {1}".format(X_sorted.shape) )
        print( X_sorted.shape)
        org_shape = X_sorted.shape
        print(org_shape)
        print("X after sorted: ".format(X_sorted))
        X_flatten = tf.reshape(X_sorted, [-1])
        print(X_flatten.shape)

        y, inverse_idx, counts = tf.unique_with_counts(X_flatten, out_idx=tf.int64)

        indices = []
        for item in y.numpy().tolist():  # y.detach().numpy().tolist():  # y.numpy().tolist():
            print("item in y：")
            print(item)
            indices.append(np.argmax(X == item))
        print("indices: ")
        print(indices)

        if if_sorted == 1:
            inverse_indices = []
            for item in X.numpy().tolist():  # X.detach().numpy().tolist():  # X.numpy().tolist():
                inverse_indices.append(np.argmax(y == item))
        else:
            inverse_indices = inverse_idx
        print("inverse_indices: ")
        print(inverse_indices)
        # tf.convert_to_tensor( value, dtype=None, dtype_hint=None, name=None)
        s = tf.convert_to_tensor(y)
        print(s)
        tf.reshape(s,[3,3])
        return tf.reshape(tf.convert_to_tensor(y), ), tf.convert_to_tensor(indices, dtype=tf.int64), \
               tf.convert_to_tensor(inverse_indices, dtype=tf.int64), tf.convert_to_tensor(counts)
