import copy

import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Reshape")
@tf_func(tf.reshape)
class Reshape(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    tensor = kwargs["tensor_dict"][node.inputs[0]]
    if cls.SINCE_VERSION == 1:
      shape = tf.constant(node.attrs["shape"], dtype=tf.int64)
    else:  # since_version >= 5
      shape = tf.cast(kwargs["tensor_dict"][node.inputs[1]], tf.int64)
      if cls.SINCE_VERSION >= 14:
        if node.attrs.get("allowzero", 0) == 1:
           return [
             cls.make_tensor_from_onnx_node(node, **kwargs)
           ]

    input_shape = tf.shape(tensor, out_type=tf.int64)

    # Extract indicies of the shape parameter where
    # a copy from the original dimension size is needed.
    sparse_indices = tf.where(tf.equal(shape, tf.constant(0, dtype=tf.int64)))
    copy_indices = tf.squeeze(sparse_indices, -1)

    indices_gathered = tf.gather(input_shape, copy_indices)
    indices_scattered = tf.sparse.to_dense(
        tf.sparse.SparseTensor(sparse_indices, indices_gathered,
                               tf.cast(tf.shape(shape), tf.int64)))

    # Perform the copy wherever requested (wherever dim_size == 0)
    copied_shape = shape + indices_scattered
    attrs = copy.deepcopy(node.attrs)
    attrs.pop("shape", None)
    return [
        cls.make_tensor_from_onnx_node(node,
                                       inputs=[tensor, copied_shape],
                                       attrs=attrs,
                                       **kwargs)
    ]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_5(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_14(cls, node, **kwargs):
    return cls._common(node, **kwargs)
